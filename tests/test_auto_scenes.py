from __future__ import annotations

from pathlib import Path

from dsense.autotest import validate_dataset
from dsense.cli import build_parser
from dsense.manifest import project_path
from dsense.scenarios import SCENARIO_GROUPS, Scenario, all_scenarios
from dsense.workloads import valid_workload_ids


def test_scenario_groups_have_required_metadata():
    labels = []
    for group, scenarios in SCENARIO_GROUPS.items():
        assert scenarios
        for scenario in scenarios:
            labels.append(scenario.label)
            assert scenario.mode == group
            assert scenario.duration > 0
            assert scenario.pre_roll >= 0
            assert scenario.action_seconds >= 0
            assert scenario.post_roll >= 0
            assert scenario.description
            assert scenario.notes
            assert scenario.duration == scenario.pre_roll + scenario.action_seconds + scenario.post_roll

    assert len(labels) == len(set(labels))


def test_automatable_scenarios_have_valid_workload_ids():
    workloads = valid_workload_ids()
    for scenario in all_scenarios():
        if scenario.automatable and scenario.workload is not None:
            assert scenario.workload in workloads
        if scenario.mode == "user":
            assert scenario.manual


def test_auto_scenes_records_short_baseline(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scenario = Scenario(
        "baseline_test_short",
        0.3,
        "Short baseline test.",
        "No action.",
        0.1,
        0.1,
        0.1,
        mode="baseline",
        manual=False,
    )
    monkeypatch.setitem(SCENARIO_GROUPS, "baseline", [scenario])

    args = build_parser().parse_args(["auto-scenes", "base", "--group", "baseline", "--include", "baseline_test_short", "--tick-hz", "10", "--yes"])
    args.func(args)

    root = project_path("base")
    scenes = sorted((root / "scenes").glob("scene_*/scene.json"))
    assert len(scenes) == 1
    result = validate_dataset("base")
    assert result.error_count == 0
    assert result.valid_scenes == 1


def test_auto_scenes_records_short_activity(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scenario = Scenario(
        "activity_test_noop",
        0.3,
        "Short activity test.",
        "No-op workload.",
        0.1,
        0.1,
        0.1,
        mode="activity",
        manual=False,
        workload="noop",
    )
    monkeypatch.setitem(SCENARIO_GROUPS, "activity", [scenario])

    args = build_parser().parse_args(["auto-scenes", "base", "--group", "activity", "--include", "activity_test_noop", "--tick-hz", "10", "--yes"])
    args.func(args)

    result = validate_dataset("base")
    assert result.error_count == 0
    assert result.valid_scenes == 1
    scene = next((project_path("base") / "scenes").glob("scene_*/scene.json"))
    assert "activity_test_noop" in scene.read_text(encoding="utf-8")
