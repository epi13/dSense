from __future__ import annotations

from pathlib import Path

from dsense.autotest import validate_dataset
from dsense.baseline import load_project_baseline
from dsense.baseline_suite import baseline_suite_report_path, plan_baseline_suite, run_baseline_suite
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
