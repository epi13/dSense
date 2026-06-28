from __future__ import annotations

from pathlib import Path

from dsense.autotest import validate_dataset
from dsense.baseline import load_project_baseline
from dsense.baseline_suite import baseline_suite_report_path, count_baseline_suite_scenes, ensure_startup_baseline_suite, plan_baseline_suite, run_baseline_suite
from dsense.cli import build_parser


def test_baseline_suite_dry_plan_reaches_target_count():
    plan = plan_baseline_suite(target_scenes=12, seed=1, duration=0.05)

    assert plan["planned_scene_count"] == 12
    assert len(plan["scenarios"]) == 12
    assert set(plan["categories"])
    assert all(str(item["label"]).startswith("baseline_") for item in plan["scenarios"])


def test_baseline_suite_include_exclude_categories():
    plan = plan_baseline_suite(target_scenes=8, categories=["idle", "cpu", "disk"], exclude_categories=["cpu"], seed=2)

    categories = {item["category"] for item in plan["scenarios"]}
    assert categories <= {"idle", "disk"}
    assert "cpu" not in categories


def test_baseline_suite_seed_is_deterministic():
    left = plan_baseline_suite(target_scenes=15, seed=42)
    right = plan_baseline_suite(target_scenes=15, seed=42)

    assert [item["label"] for item in left["scenarios"]] == [item["label"] for item in right["scenarios"]]


def test_baseline_suite_network_and_heavy_disabled_by_default():
    plan = plan_baseline_suite(target_scenes=40, seed=3)

    assert plan["network_enabled"] is False
    assert plan["heavy_enabled"] is False
    assert "network" not in {item["category"] for item in plan["scenarios"]}
    assert not any(item.get("workload") == "cpu_heavy" for item in plan["scenarios"])


def test_baseline_suite_tiny_run_writes_report_and_validates(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    report = run_baseline_suite(
        "base",
        target_scenes=3,
        categories=["idle"],
        seed=4,
        duration=0.03,
        tick_hz=10,
        linux=False,
        assume_yes=True,
    )

    assert report["actual_scene_count"] == 3
    assert baseline_suite_report_path("base").exists()
    assert validate_dataset("base").error_count == 0
    model = load_project_baseline("base")
    assert model is not None
    assert model.scene_count >= 3


def test_startup_baseline_suite_fills_to_target_once(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    first = ensure_startup_baseline_suite("base", target_scenes=3, duration=0.03, tick_hz=10, linux=False)
    second = ensure_startup_baseline_suite("base", target_scenes=3, duration=0.03, tick_hz=10, linux=False)

    assert first["status"] == "recorded"
    assert first["recorded"] == 3
    assert second["status"] == "reused"
    assert count_baseline_suite_scenes("base") == 3


def test_baseline_suite_emits_recording_progress(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    updates = []

    run_baseline_suite(
        "base",
        target_scenes=1,
        duration=0.03,
        tick_hz=10,
        linux=False,
        assume_yes=True,
        progress_callback=updates.append,
    )

    assert updates
    assert any(update["phase"] == "baseline_suite" and update["status"] == "recording" for update in updates)
    assert any(update["status"] == "recorded" for update in updates)


def test_startup_baseline_suite_fills_missing_with_label_offset(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_startup_baseline_suite("base", target_scenes=2, duration=0.03, tick_hz=10, linux=False)

    result = ensure_startup_baseline_suite("base", target_scenes=4, duration=0.03, tick_hz=10, linux=False)
    labels = sorted(path.read_text(encoding="utf-8") for path in (tmp_path / "datasets" / "base" / "scenes").glob("scene_*/scene.json"))

    assert result["recorded"] == 2
    assert count_baseline_suite_scenes("base") == 4
    assert any("_003" in label for label in labels)
    assert any("_004" in label for label in labels)


def test_baseline_suite_cli_parser_accepts_options():
    args = build_parser().parse_args([
        "baseline-suite",
        "base",
        "--target-scenes",
        "5",
        "--repeat",
        "2",
        "--categories",
        "idle,cpu",
        "--exclude-categories",
        "cpu",
        "--seed",
        "7",
        "--dry-run",
        "--yes",
    ])

    assert args.target_scenes == 5
    assert args.repeat == 2
    assert args.categories == "idle,cpu"
    assert args.exclude_categories == "cpu"
    assert args.seed == 7
    assert args.dry_run is True


def test_tui_parser_accepts_startup_suite_flags():
    args = build_parser().parse_args([
        "tui",
        "base",
        "--no-startup-suite",
        "--startup-suite-target",
        "7",
        "--startup-suite-duration",
        "0.1",
        "--startup-suite-seed",
        "9",
        "--no-startup-suite-linux",
    ])

    assert args.no_startup_suite is True
    assert args.startup_suite_target == 7
    assert args.startup_suite_duration == 0.1
    assert args.startup_suite_seed == 9
    assert args.startup_suite_linux is False
