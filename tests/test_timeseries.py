from __future__ import annotations

from pathlib import Path

from dsense.manifest import DEFAULT_PROJECT
from dsense.timeseries import (
    load_project_timeseries,
    predict_scene_timeseries,
    timeseries_path,
    train_and_save_project_timeseries,
)


def test_train_project_timeseries_persists_temporal_profiles(sample_dataset):
    model = train_and_save_project_timeseries(DEFAULT_PROJECT)
    loaded = load_project_timeseries(DEFAULT_PROJECT)

    assert timeseries_path(DEFAULT_PROJECT).exists()
    assert loaded is not None
    assert model.scene_count == 2
    assert model.label_counts["baseline_idle"] == 1
    assert "sleep_drift_ns" in model.sequence_channels
    assert model.label_profiles


def test_predict_scene_timeseries_returns_label_confidence_and_channels(sample_dataset: Path):
    model = train_and_save_project_timeseries(DEFAULT_PROJECT)
    prediction = predict_scene_timeseries(model, sample_dataset / "scenes" / "scene_000001" / "preview.csv")

    assert prediction["label"] in {"baseline_idle", "typing_burst"}
    assert 0.0 <= float(prediction["confidence"]) <= 1.0
    assert prediction["sequence_channels"]
