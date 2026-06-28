import csv

from dsense.baseline import load_project_baseline, score_against_baseline, train_and_save_project_baseline
from dsense.manifest import DEFAULT_PROJECT, init_project
from dsense.utils.files import write_json


def _write_baseline_scene(root, scene_id, values):
    scene_dir = root / "scenes" / scene_id
    scene_dir.mkdir()
    write_json(scene_dir / "scene.json", {"scene_id": scene_id, "label": "baseline_idle", "accepted": True})
    with (scene_dir / "preview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags"])
        writer.writeheader()
        for tick, drift in enumerate(values):
            writer.writerow({"tick": tick, "t_ns": tick, "dt_ns": 10_000_000, "sleep_drift_ns": drift, "process_ns_estimate": 8_000, "quality_flags": 0})


def test_train_and_score_baseline_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_baseline_scene(root, "scene_000001", [100, 110, 90, 100, 105, 95])

    model = train_and_save_project_baseline(DEFAULT_PROJECT)
    loaded = load_project_baseline(DEFAULT_PROJECT)
    score = score_against_baseline({"sleep_drift_ns": 5000}, loaded)

    assert model.scene_count == 1
    assert loaded is not None
    assert "sleep_drift_ns" in loaded.channels
    assert "sleep_drift_ns_slope" in loaded.feature_manifest["features"]
    assert score["status"] == "anomaly"
