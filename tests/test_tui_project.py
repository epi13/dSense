from dsense.cli import build_parser
from dsense.classifier import SceneClassifierModel
from dsense.manifest import DEFAULT_PROJECT, init_project
from dsense.scenarios import SCENARIO_GROUPS
from dsense.tui import (
    TABS,
    clip_text,
    classifier_summary_lines,
    load_project_scenes,
    scene_detail_lines,
    scheduled_scene_events,
    summarize_scene_counts,
    system_event_marker,
    tab_index_delta,
    wrap_text,
)
from dsense.utils.files import write_json


def test_tui_command_defaults_to_base_project():
    args = build_parser().parse_args(["tui"])
    assert args.project_name == DEFAULT_PROJECT


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
    assert TABS == ["Record", "Scenes", "Channels", "Learn", "Classify", "Watcher", "Orbiters", "Transfer", "Validate", "Help"]


def test_tab_index_delta_wraps_next_and_previous():
    assert tab_index_delta(0, 1) == 1
    assert tab_index_delta(len(TABS) - 1, 1) == 0
    assert tab_index_delta(0, -1) == len(TABS) - 1


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
