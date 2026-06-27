import csv

from dsense.classifier import classifier_path, load_project_classifier, train_and_save_project_classifier
from dsense.event_detector import HeuristicEventDetector
from dsense.manifest import DEFAULT_PROJECT, init_project
from dsense.utils.files import write_json


def _write_scene(root, scene_id, label, drift_values):
    scene_dir = root / "scenes" / scene_id
    scene_dir.mkdir()
    write_json(scene_dir / "scene.json", {
        "scene_id": scene_id,
        "label": label,
        "accepted": True,
        "quality": {"confidence": 1.0},
    })
    with (scene_dir / "preview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags"])
        writer.writeheader()
        for tick, drift in enumerate(drift_values):
            writer.writerow({
                "tick": tick,
                "t_ns": tick,
                "dt_ns": 10_000_000,
                "sleep_drift_ns": drift,
                "process_ns_estimate": 8_000,
                "quality_flags": 0,
            })


def test_train_project_classifier_persists_baseline_profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_scene(root, "scene_000001", "baseline_idle", [100, 110, 90, 105, 95, 100])
    _write_scene(root, "scene_000002", "user_interaction_approach", [100, 110, 5000, 105, 95, 100])

    model = train_and_save_project_classifier(DEFAULT_PROJECT)
    loaded = load_project_classifier(DEFAULT_PROJECT)

    assert classifier_path(DEFAULT_PROJECT).exists()
    assert loaded is not None
    assert model.scene_count == 2
    assert model.baseline_scene_count == 1
    assert "sleep_drift_ns" in model.detector_baseline
    assert model.label_counts["baseline_idle"] == 1


def test_detector_uses_learned_baseline_before_local_window_warms():
    detector = HeuristicEventDetector(
        tick_hz=20,
        learned_baseline={"sleep_drift_ns": {"center": 100.0, "mad": 10.0}},
        threshold=5.0,
        warmup_samples=1,
    )

    events = detector.update({
        "tick": 1,
        "elapsed_ms": 50,
        "dt_ns": 10_000_000,
        "sleep_drift_ns": 5000,
        "process_ns_estimate": 8_000,
    })

    assert events
    assert events[0]["event"] == "heuristic_signal_spike"
    assert events[0]["channel"] == "sleep_drift_ns"
