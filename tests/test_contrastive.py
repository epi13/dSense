from __future__ import annotations

import csv
import sys
from pathlib import Path

from dsense.contrastive import (
    contrastive_path,
    extract_contrastive_features,
    label_to_scene_family,
    load_project_contrastive,
    predict_scene_contrastive,
    train_and_save_project_contrastive,
)
from dsense.manifest import DEFAULT_PROJECT
from dsense.utils.files import write_json


def test_extract_contrastive_features_returns_temporal_features():
    rows = [
        {"sleep_drift_ns": 1.0, "cpu_load_ppm": 10.0},
        {"sleep_drift_ns": 3.0, "cpu_load_ppm": 12.0},
        {"sleep_drift_ns": 2.0, "cpu_load_ppm": 11.0},
        {"sleep_drift_ns": 5.0, "cpu_load_ppm": 15.0},
    ]

    features = extract_contrastive_features(rows)

    assert features
    assert "sleep_drift_ns_median" in features
    assert "sleep_drift_ns_roughness" in features
    assert "sleep_drift_ns_window_1_median_delta" in features
    assert any(key.endswith("_mean_ratio") for key in features)


def test_contrastive_family_mapping_is_broad_and_deterministic():
    assert label_to_scene_family("baseline_idle") == "baseline"
    assert label_to_scene_family("activity_cpu_heavy") == "machine_activity"
    assert label_to_scene_family("person_walks_front") == "user_presence"
    assert label_to_scene_family("typing_burst") == "user_presence"
    assert label_to_scene_family("mystery") == "unknown"


def test_train_contrastive_skips_rejected_and_persists(sample_dataset: Path):
    _write_preview_scene(sample_dataset, "scene_000003", "person_walks_front", [1000, 1200, 900], accepted=False)

    model = train_and_save_project_contrastive(DEFAULT_PROJECT)
    loaded = load_project_contrastive(DEFAULT_PROJECT)

    assert contrastive_path(DEFAULT_PROJECT).exists()
    assert loaded is not None
    assert model.scene_count == 2
    assert "person_walks_front" not in model.label_counts
    assert model.family_counts["baseline"] == 1
    assert model.family_counts["user_presence"] == 1
    assert model.label_profiles
    assert model.family_profiles


def test_predict_contrastive_returns_expected_keys(sample_dataset: Path):
    model = train_and_save_project_contrastive(DEFAULT_PROJECT)

    prediction = predict_scene_contrastive(model, sample_dataset / "scenes" / "scene_000001" / "preview.csv")

    assert {
        "family",
        "label",
        "confidence",
        "family_distance",
        "label_distance",
        "nearest_family_distances",
        "nearest_label_distances",
        "contributions",
        "sequence_channels",
        "backend",
    } <= set(prediction)
    assert prediction["backend"] == "profile"
    assert 0.0 <= float(prediction["confidence"]) <= 1.0


def test_tiny_or_missing_contrastive_model_returns_unknown(sample_dataset: Path):
    missing = predict_scene_contrastive(None, sample_dataset / "scenes" / "scene_000001" / "preview.csv")

    assert missing["family"] == "unknown"
    assert missing["label"] == "unknown"
    assert missing["confidence"] == 0.0


def test_torch_backend_reports_unavailable_without_required_dependency(sample_dataset: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    model = train_and_save_project_contrastive(DEFAULT_PROJECT, backend="torch_tcn")

    assert model.backend == "profile"
    assert any("PyTorch is not installed" in warning for warning in model.feature_manifest.get("training_warnings", []))


def _write_preview_scene(root: Path, scene_id: str, label: str, drift_values: list[int], *, accepted: bool) -> None:
    scene_dir = root / "scenes" / scene_id
    scene_dir.mkdir()
    with (scene_dir / "preview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tick", "t_ns", "sleep_drift_ns", "cpu_load_ppm"])
        writer.writeheader()
        for tick, drift in enumerate(drift_values):
            writer.writerow({"tick": tick, "t_ns": tick * 100, "sleep_drift_ns": drift, "cpu_load_ppm": 10_000 + tick})
    write_json(scene_dir / "scene.json", {"scene_id": scene_id, "label": label, "accepted": accepted, "created_utc": "2026-06-27T00:00:00Z"})
