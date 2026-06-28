from __future__ import annotations

from dsense import council
from dsense.council import classify_with_council, intelligence_state_path, run_intelligence_update
from dsense.manifest import DEFAULT_PROJECT


def test_run_intelligence_update_writes_state(sample_dataset):
    state = run_intelligence_update(DEFAULT_PROJECT, run_watchers=False, run_orbiters=False, run_transfer=True)

    assert intelligence_state_path(DEFAULT_PROJECT).exists()
    assert state["format"] == "dsense-intelligence-state-v1"
    assert state["models"]["baseline"]["scene_count"] == 1
    assert state["models"]["classifier"]["scene_count"] == 2
    assert state["models"]["timeseries"]["scene_count"] == 2
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
    assert "baseline_anomaly" in result
    assert result["agreement"] in {"unknown", "low", "high"}
