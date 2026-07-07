from __future__ import annotations

from dsense import council
from dsense.council import build_council_summary, classify_with_council, intelligence_state_path, run_intelligence_update
from dsense.manifest import DEFAULT_PROJECT


def test_run_intelligence_update_writes_state(sample_dataset):
    state = run_intelligence_update(DEFAULT_PROJECT, run_watchers=False, run_orbiters=False, run_transfer=True)

    assert intelligence_state_path(DEFAULT_PROJECT).exists()
    assert state["format"] == "dsense-intelligence-state-v1"
    assert state["models"]["baseline"]["scene_count"] == 1
    assert state["models"]["classifier"]["scene_count"] == 2
    assert state["models"]["timeseries"]["scene_count"] == 2
    assert state["models"]["contrastive"]["scene_count"] == 2
    assert "overall_confidence" in state["council"]


def test_council_continues_when_one_step_fails(sample_dataset, monkeypatch):
    def fail_timeseries(project_name):
        raise ValueError("time-series unavailable")

    monkeypatch.setattr(council, "train_and_save_project_timeseries", fail_timeseries)

    state = run_intelligence_update(DEFAULT_PROJECT, run_watchers=False, run_orbiters=False, run_transfer=False)

    assert state["status"] == "failed"
    assert any(step["name"] == "train_timeseries" and step["status"] == "failed" for step in state["steps"])
    assert intelligence_state_path(DEFAULT_PROJECT).exists()


def test_classify_with_council_combines_local_layers(sample_dataset):
    run_intelligence_update(DEFAULT_PROJECT, run_watchers=False, run_orbiters=False, run_transfer=False)

    result = classify_with_council(DEFAULT_PROJECT, sample_dataset / "scenes" / "scene_000001")

    assert "deterministic_classifier" in result
    assert "time_series_classifier" in result
    assert "contrastive_temporal" in result
    assert "baseline_anomaly" in result
    assert result["agreement"] in {"unknown", "low", "medium", "high"}


def test_council_groups_user_label_variants_for_repeatability():
    summary = build_council_summary(
        DEFAULT_PROJECT,
        {
            "baseline": {"scene_count": 3},
            "classifier": {
                "scene_count": 5,
                "label_counts": {
                    "baseline_idle": 3,
                    "Approach": 1,
                    "user_approach_from_left": 1,
                    "user_approach_from_right": 1,
                },
            },
            "timeseries": {"scene_count": 5, "label_counts": {}},
            "watcher": {"event_count": 1},
            "orbiters": {"summary_count": 1},
            "evaluation": {"confusion_matrix": {"accuracy": 1.0}, "baseline_drift": {"max_drift": 0.0}},
        },
    )

    assert "not enough repeated user labels" not in summary["warnings"]
    assert "approach" not in summary["weak_labels"]


def test_council_recommends_auto_scenes_for_short_activity_controls():
    summary = build_council_summary(
        DEFAULT_PROJECT,
        {
            "baseline": {"scene_count": 3},
            "classifier": {"scene_count": 4, "label_counts": {"baseline_idle": 3, "activity_cpu_heavy": 1}},
            "timeseries": {"scene_count": 4, "label_counts": {}},
            "watcher": {"event_count": 1},
            "orbiters": {"summary_count": 1},
            "evaluation": {"confusion_matrix": {"accuracy": 1.0}, "baseline_drift": {"max_drift": 0.0}},
        },
    )

    assert "not enough repeated automatic control labels" in summary["warnings"]
    assert any("auto-scenes base --include activity_cpu_heavy --repeat 2 --yes" in item for item in summary["recommendations"])
    assert not any("record 3 more takes of label activity_cpu_heavy" in item for item in summary["recommendations"])
