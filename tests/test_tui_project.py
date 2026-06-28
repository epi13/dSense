import tomllib
from pathlib import Path

from dsense import cli, tui_app
from dsense.cli import build_parser
from dsense.classifier import SceneClassifierModel
from dsense.manifest import DEFAULT_PROJECT, init_project
from dsense.tui_state import AppState, CaptureConfig, JobState, RecordingState
from dsense.tui_jobs import TuiJobManager
from dsense.tui_render import compact_live_observation_lines, live_footer_text, live_observation_lines, robust_channel_score, sparkline, value_channel_id
from dsense.tui_render import evaluation_repeatability_lines, labels_needing_more_takes, useful_channel_lines
from dsense.scenarios import SCENARIO_GROUPS
from dsense.tui import (
    TABS,
    clip_text,
    classifier_summary_lines,
    load_project_scenes,
    scene_detail_lines,
    scheduled_scene_events,
    channel_state_label,
    format_metric_value,
    summarize_scene_counts,
    system_event_marker,
    tab_index_delta,
    wrap_text,
)
from dsense.utils.files import write_json


def test_tui_command_defaults_to_base_project():
    args = build_parser().parse_args(["tui"])
    assert args.project_name == DEFAULT_PROJECT


def test_tui_optional_dependencies_are_declared():
    data = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["optional-dependencies"]["tui"] == ["textual", "rich"]
    assert "pytest>=7" in data["project"]["optional-dependencies"]["dev"]


def test_tui_app_falls_back_to_curses_when_textual_missing(monkeypatch):
    called = {}

    monkeypatch.setattr(tui_app, "_textual_available", lambda: False)
    def fake_curses(config):
        called["project"] = config.project_name
        return []

    monkeypatch.setattr(tui_app, "_run_curses_tui", fake_curses)

    result = tui_app.run_tui(CaptureConfig(project_name="base"))

    assert result == []
    assert called["project"] == "base"


def test_tui_app_prefers_curses_when_startup_pipeline_enabled(monkeypatch):
    called = {}

    monkeypatch.setattr(tui_app, "_textual_available", lambda: True)

    def fake_curses(config):
        called["backend"] = "curses"
        called["project"] = config.project_name
        return []

    monkeypatch.setattr(tui_app, "_run_curses_tui", fake_curses)
    monkeypatch.setattr(tui_app, "_run_textual_tui", lambda config: called.setdefault("backend", "textual") or [])

    tui_app.run_tui(CaptureConfig(project_name="base", auto_baseline_policy="auto", startup_suite_enabled=True))

    assert called == {"backend": "curses", "project": "base"}


def test_tui_safe_parser_and_startup_disable_flags_are_accepted():
    args = build_parser().parse_args(["tui", "base", "--no-startup-intelligence", "--no-startup-watchers", "--no-startup-orbiters", "--no-startup-training"])
    safe = build_parser().parse_args(["tui-safe", "base"])

    assert args.no_startup_intelligence is True
    assert args.no_startup_watchers is True
    assert args.no_startup_orbiters is True
    assert args.no_startup_training is True
    assert safe.project_name == "base"


def test_tui_state_dataclasses_hold_runtime_state():
    config = CaptureConfig(project_name="base", channel_groups=["portable", "linux"])
    app_state = AppState(config=config, jobs=JobState(validation_summary="ready"), recording=RecordingState(latest={"tick": 1}))

    assert app_state.config.channel_groups == ["portable", "linux"]
    assert app_state.jobs.validation_summary == "ready"
    assert app_state.recording.latest["tick"] == 1


def test_tui_job_manager_runs_and_logs_jobs(tmp_path, monkeypatch):
    import time

    monkeypatch.chdir(tmp_path)
    manager = TuiJobManager("base")

    manager.start("train baseline", lambda update, cancel: update("working") or "done detail")
    for _ in range(100):
        jobs = manager.snapshot()
        if jobs and jobs[-1].status == "done":
            break
        time.sleep(0.001)
    else:
        raise AssertionError("job did not finish")

    job = manager.snapshot()[-1]
    assert job.name == "train baseline"
    assert job.detail == "done detail"
    assert (tmp_path / "datasets" / "base" / "jobs" / "events.jsonl").exists()


def test_tui_job_manager_records_cancel_request(tmp_path, monkeypatch):
    import time

    monkeypatch.chdir(tmp_path)
    manager = TuiJobManager("base")

    def slow_job(update, cancel):
        while not cancel.is_set():
            time.sleep(0.001)
        return "cancelled cooperatively"

    manager.start("validate dataset", slow_job)
    assert manager.cancel_running() is True
    for _ in range(100):
        jobs = manager.snapshot()
        if jobs and jobs[-1].status == "cancelled":
            break
        time.sleep(0.001)
    else:
        raise AssertionError("job did not cancel")

    assert manager.snapshot()[-1].cancel_requested is True


def test_tui_parser_accepts_channel_groups():
    args = build_parser().parse_args(["tui", "base", "--channels", "portable,linux"])

    assert args.channels == "portable,linux"


def test_live_cli_parser_accepts_live_and_start_tabs():
    parser = build_parser()
    live = parser.parse_args(["live", "base"])
    tui_live = parser.parse_args(["tui", "base", "--live"])
    start_live = parser.parse_args(["tui", "base", "--start-tab", "live"])
    start_capture = parser.parse_args(["tui", "base", "--start-tab", "capture"])

    assert live.project_name == "base"
    assert tui_live.live is True
    assert start_live.start_tab == "live"
    assert start_capture.start_tab == "capture"


def test_startup_performance_flags_parse():
    parser = build_parser()
    fast = parser.parse_args(["tui", "base", "--fast-start"])
    force = parser.parse_args(["tui", "base", "--force-startup-update", "--no-startup-orbiters"])
    alias = parser.parse_args(["tui-fast", "base"])

    assert fast.fast_start is True
    assert force.force_startup_update is True
    assert force.no_startup_orbiters is True
    assert alias.project_name == "base"


def test_intelligence_cli_commands_parse():
    parser = build_parser()

    assert parser.parse_args(["train-timeseries", "base"]).project_name == "base"
    assert parser.parse_args(["update-intelligence", "base", "--no-watchers"]).no_watchers is True
    assert parser.parse_args(["council-status", "base"]).project_name == "base"


def test_scene_tui_passes_channel_groups_to_capture_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_run_tui_config(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "_run_tui_config", fake_run_tui_config)
    args = build_parser().parse_args([
        "scene",
        "base",
        "--label",
        "test_scene",
        "--duration",
        "10",
        "--channels",
        "portable,linux",
        "--tui",
    ])

    cli.cmd_scene(args)

    assert captured["channel_groups"] == ["portable", "linux"]
    assert captured["auto_baseline_policy"] == "off"
    assert captured["startup_suite_enabled"] is False


def test_tui_command_passes_channel_groups_to_capture_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = {}

    monkeypatch.setattr(cli, "_run_tui_config", lambda **kwargs: captured.update(kwargs) or [])
    args = build_parser().parse_args([
        "tui",
        "base",
        "--channels",
        "portable,linux",
        "--no-auto-baseline",
        "--no-startup-suite",
    ])

    cli.cmd_tui(args)

    assert captured["channel_groups"] == ["portable", "linux"]
    assert captured["auto_baseline_policy"] == "off"
    assert captured["startup_suite_enabled"] is False
    assert captured["startup_suite_target"] == 200
    assert captured["start_tab"] == "live"


def test_tui_no_startup_intelligence_disables_startup_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = {}

    monkeypatch.setattr(cli, "_run_tui_config", lambda **kwargs: captured.update(kwargs) or [])
    args = build_parser().parse_args(["tui", "base", "--no-startup-intelligence"])

    cli.cmd_tui(args)

    assert captured["startup_intelligence"] is False
    assert captured["startup_watchers"] is False
    assert captured["startup_orbiters"] is False
    assert captured["startup_training"] is False
    assert captured["auto_baseline_policy"] == "off"
    assert captured["startup_suite_enabled"] is False


def test_load_project_scenes_reads_existing_project_scenes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    scene_dir = root / "scenes" / "scene_000001"
    scene_dir.mkdir()
    write_json(scene_dir / "scene.json", {
        "scene_id": "scene_000001",
        "label": "baseline_idle",
        "accepted": True,
        "quality": {"confidence": 1.0},
    })

    scenes = load_project_scenes(DEFAULT_PROJECT)

    assert [scene["scene_id"] for scene in scenes] == ["scene_000001"]
    assert scenes[0]["label"] == "baseline_idle"


def test_tui_scenario_groups_include_baseline_activity_and_user_presets():
    assert {"baseline", "activity", "user"} <= set(SCENARIO_GROUPS)
    assert SCENARIO_GROUPS["baseline"]
    assert SCENARIO_GROUPS["activity"]
    assert SCENARIO_GROUPS["user"]


def test_tui_tabs_include_expected_sections():
    assert TABS == ["Live", "Sense Radar", "Council", "Capture", "Scenes", "Evaluation", "Watchers", "Orbiters", "Transfer", "Settings"]


def test_tui_help_points_to_update_intelligence_action():
    assert "Council" in TABS
    assert any("Press u" in line for line in classifier_summary_lines(None, True))
    footer = live_footer_text()
    assert "m mark" in footer
    assert "r record" in footer
    assert "u update intelligence" in footer
    assert "s snapshot" in footer


def test_tab_index_delta_wraps_next_and_previous():
    assert tab_index_delta(0, 1) == 1
    assert tab_index_delta(len(TABS) - 1, 1) == 0
    assert tab_index_delta(0, -1) == len(TABS) - 1


def test_live_rendering_handles_small_width():
    lines = compact_live_observation_lines(None, 30)
    full = live_observation_lines(None, 30)

    assert lines
    assert full
    assert all(len(line) <= 30 for line in full)


def test_wrap_text_handles_empty_and_long_notes():
    assert wrap_text("", 20) == ["(no notes)"]
    lines = wrap_text("alpha beta gamma delta", 10)
    assert lines == ["alpha beta", "gamma", "delta"]
    assert all(len(line) <= 10 for line in wrap_text("averyveryverylongword", 6))


def test_clip_text_keeps_record_fields_inside_panel_width():
    assert clip_text("short", 10) == "short"
    assert clip_text("this is much too long", 12) == "this is m..."
    assert len(clip_text("this is much too long", 12)) <= 12

    prefix = f"{'>':1} {'Notes':<18} "
    value_width = 35 - len(prefix)
    lines = [clip_text(prefix + part, 35) for part in wrap_text("Start outside the sensing area on the left, enter during the action window.", value_width)]
    assert lines
    assert all(len(line) <= 35 for line in lines)


def test_scene_detail_lines_handle_missing_and_long_notes_separately():
    lines = scene_detail_lines({
        "scene_id": "scene_000001",
        "label": "baseline_idle",
        "duration_ms": 1000,
        "tick_hz": 10,
        "accepted": True,
        "quality": {"confidence": 1.0, "actual_frames": 10, "expected_frames": 10},
    })

    assert any("scene_000001" in line for line in lines)
    assert any("baseline_idle" in line for line in lines)


def test_summarize_scene_counts_splits_baseline_and_user_labels():
    counts = summarize_scene_counts([
        {"label": "baseline_idle"},
        {"label": "baseline_cpu_light"},
        {"label": "person_walks_front_left_to_right"},
        {"label": "user_interaction_approach"},
        {"label": "custom_label"},
    ])

    assert counts == {"baseline": 2, "user": 2, "other": 1}


def test_scheduled_scene_events_match_capture_windows():
    events = scheduled_scene_events(duration=10, pre_roll=2, action=5, post_roll=3)

    assert events == [
        {"t_ms": 0, "event": "scene_start", "source": "system"},
        {"t_ms": 2000, "event": "action_start", "source": "system"},
        {"t_ms": 7000, "event": "action_end", "source": "system"},
        {"t_ms": 10000, "event": "scene_end", "source": "system"},
    ]
    assert [system_event_marker(event["event"]) for event in events] == ["S", "A", "E", "X"]


def test_classifier_summary_lines_show_model_status():
    model = SceneClassifierModel(
        project_name=DEFAULT_PROJECT,
        trained_utc="2026-06-27T12:00:00Z",
        scene_count=41,
        baseline_scene_count=30,
        label_counts={"baseline_idle": 30, "typing_burst": 2},
        detector_baseline={"dt_ns": {"center": 1.0, "mad": 1.0}},
        label_profiles={},
    )

    lines = classifier_summary_lines(model, auto_detect=True)

    assert lines[0] == "Active"
    assert "trained scenes 41" in lines[1]
    assert "baseline 30" in lines[1]
    assert "using learned baseline" in lines[2]
    assert "dt_ns" in lines[3]


def test_classifier_summary_lines_handle_missing_model():
    lines = classifier_summary_lines(None, auto_detect=True)

    assert lines[0] == "No classifier trained yet"


def test_live_telemetry_helpers_format_values_and_channel_states():
    assert format_metric_value(None) == "-"
    assert format_metric_value(12) == "12"
    assert format_metric_value(0.012345) == "0.0123"

    assert channel_state_label({"available": True, "sampled": True}) == "sampled"
    assert channel_state_label({"available": True, "stale": True}) == "stale"
    assert channel_state_label({"available": True}) == "idle"
    assert channel_state_label({"available": False, "unavailable": True}) == "offline"


def test_channel_dashboard_helpers_score_sparkline_and_map_values():
    assert value_channel_id("battery_percent") == "power_state"
    assert value_channel_id("disk_stat_latency_ns") == "disk_latency"
    assert value_channel_id("custom_value") == "custom_value"

    assert robust_channel_score("cpu_load_ppm", 130.0, {"cpu_load_ppm": {"center": 100.0, "mad": 10.0}}) == 3.0
    assert robust_channel_score("custom", 10.0, None, [1, 2, 2, 3, 10]) > 0

    line = sparkline([1, 2, 3, 4], width=4)
    assert len(line) == 4
    assert sparkline([5, 5, 5], width=5) == "─" * 5


def test_evaluation_tab_helpers_surface_truth_metrics():
    report = {
        "within_label_similarity": {"overall": 0.82, "labels": {"door_open_close": 0.5}},
        "between_label_distance": {"average": 1.74},
        "confusion_matrix": {
            "accuracy": 0.68,
            "matrix": {
                "phone_near_computer": {"typing_burst": 1, "phone_near_computer": 1},
                "door_open_close": {"door_open_close": 2},
            },
        },
        "baseline_drift": {"max_drift": 0.31},
        "label_counts": {"walk_left_to_right": 1, "phone_near_computer": 2, "door_open_close": 2},
        "channel_usefulness_ranking": [
            {"channel": "disk_stat_latency_ns", "score": 4.12},
            {"channel": "cpu_load_ppm", "score": 2.77},
        ],
    }

    assert evaluation_repeatability_lines(report) == [
        "within-label similarity: 0.82",
        "between-label distance: 1.74",
        "leave-one-out accuracy: 68%",
        "baseline drift max: 0.31",
    ]
    weak = labels_needing_more_takes(report)
    assert ("walk_left_to_right", "1 take") in weak
    assert ("phone_near_computer", "confused with typing_burst (1)") in weak
    assert ("door_open_close", "high variance") in weak
    assert useful_channel_lines(report, 2) == [
        "disk_stat_latency_ns       score 4.12",
        "cpu_load_ppm               score 2.77",
    ]
