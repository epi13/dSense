import csv
import hashlib
import json
from pathlib import Path

import pytest

from dsense.cli import main
from dsense.classifier import train_project_classifier, train_and_save_project_classifier
from dsense.frame import FRAME_SIZE, build_frame
from dsense.manifest import DEFAULT_PROJECT, init_project
from dsense.models.evaluation import evaluate_project_scenes, evaluation_report_path
from dsense.models.features import discover_numeric_preview_columns, summarize_preview


def _write_complete_scene(root: Path, scene_id: str, label: str, values: list[int], *, cpu_base: int = 0, valid: bool = True) -> Path:
    scene_dir = root / "scenes" / scene_id
    scene_dir.mkdir(parents=True)
    frames = b"".join(
        build_frame(tick, 1_000_000_000 + tick, 7, 0, 10_000_000, value, 8_000)
        for tick, value in enumerate(values)
    )
    if not valid:
        frames = frames[:-1]
    (scene_dir / "frames.ds64").write_bytes(frames)
    fieldnames = ["tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags", "cpu_load_ppm", "text_note"]
    with (scene_dir / "preview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tick, value in enumerate(values):
            writer.writerow({
                "tick": tick,
                "t_ns": 1_000_000_000 + tick,
                "dt_ns": 10_000_000,
                "sleep_drift_ns": value,
                "process_ns_estimate": 8_000,
                "quality_flags": 0,
                "cpu_load_ppm": cpu_base + tick,
                "text_note": "ignore-me",
            })
    events = [
        {"t_ms": 0, "event": "scene_start"},
        {"t_ms": 0, "event": "action_start"},
        {"t_ms": len(values) * 100, "event": "action_end"},
        {"t_ms": len(values) * 100, "event": "scene_end"},
    ]
    (scene_dir / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    sha = hashlib.sha256(frames).hexdigest()
    (scene_dir / "checksum.txt").write_text(f"sha256  frames.ds64  {sha}\n", encoding="utf-8")
    (scene_dir / "notes.txt").write_text("", encoding="utf-8")
    scene = {
        "scene_id": scene_id,
        "label": label,
        "created_utc": f"2026-06-27T00:00:0{scene_id[-1]}Z",
        "duration_ms": len(values) * 100,
        "tick_hz": 10,
        "frame_size_bytes": FRAME_SIZE,
        "quality": {
            "expected_frames": len(values),
            "actual_frames": len(values),
            "confidence": 1.0,
            "checksum_ok": True,
            "frame_size_valid": True,
            "jitter_ns": 0,
        },
        "accepted": True,
    }
    (scene_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    return scene_dir


def test_dynamic_numeric_preview_discovery(tmp_path):
    scene_dir = tmp_path / "scene_000001"
    scene_dir.mkdir()
    with (scene_dir / "preview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tick", "t_ns", "dt_ns", "quality_flags", "cpu_load_ppm", "note"])
        writer.writeheader()
        writer.writerow({"tick": 0, "t_ns": 1, "dt_ns": 10, "quality_flags": 0, "cpu_load_ppm": 42, "note": "idle"})

    columns = discover_numeric_preview_columns(scene_dir / "preview.csv")
    features = summarize_preview(scene_dir / "preview.csv")

    assert columns == ["cpu_load_ppm", "dt_ns"]
    assert features["cpu_load_ppm_median"] == 42
    assert "note_median" not in features


def test_classifier_uses_extra_numeric_channels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_complete_scene(root, "scene_000001", "baseline_idle", [100, 100, 100], cpu_base=10)
    _write_complete_scene(root, "scene_000002", "cpu_load_no_person", [100, 100, 100], cpu_base=900)

    model = train_project_classifier(DEFAULT_PROJECT)

    assert "cpu_load_ppm" in model.detector_baseline
    assert "cpu_load_ppm_median" in model.label_profiles["cpu_load_no_person"]


def test_evaluate_scenes_writes_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_complete_scene(root, "scene_000001", "baseline_idle", [100, 100, 100], cpu_base=10)
    _write_complete_scene(root, "scene_000002", "baseline_idle", [110, 110, 110], cpu_base=11)
    _write_complete_scene(root, "scene_000003", "typing_burst", [5000, 5100, 5200], cpu_base=100)
    _write_complete_scene(root, "scene_000004", "typing_burst", [5300, 5400, 5500], cpu_base=110)

    report = evaluate_project_scenes(DEFAULT_PROJECT)

    assert evaluation_report_path(DEFAULT_PROJECT).exists()
    assert report["label_counts"] == {"baseline_idle": 2, "typing_burst": 2}
    assert report["confusion_matrix"]["evaluated"] == 4
    assert report["channel_usefulness_ranking"]


def test_require_valid_blocks_training_before_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_complete_scene(root, "scene_000001", "baseline_idle", [100, 100], valid=False)

    with pytest.raises(SystemExit):
        main(["train-classifier", DEFAULT_PROJECT, "--require-valid"])

    assert not (root / "exports" / "classifier.json").exists()


def test_replay_classify_and_inspect_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    scene_dir = _write_complete_scene(root, "scene_000001", "baseline_idle", [100, 100, 100], cpu_base=10)
    train_and_save_project_classifier(DEFAULT_PROJECT)

    main(["classify-scene", str(scene_dir), "--project", DEFAULT_PROJECT])
    classify_out = capsys.readouterr().out
    assert '"label": "baseline_idle"' in classify_out

    main(["inspect-frame", str(scene_dir), "--tick", "1"])
    inspect_out = capsys.readouterr().out
    assert '"sequence": 1' in inspect_out
    assert '"preview"' in inspect_out

    main(["replay", str(scene_dir), "--limit", "1"])
    replay_out = capsys.readouterr().out
    assert "Replay scene_000001" in replay_out
    assert "Preview:" in replay_out
