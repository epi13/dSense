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
        writer = csv.DictWriter(handle, fieldnames=["tick", "t_ns", "dt_ns", "quality_flags", "cpu_load_ppm", "probe_acc", "note"])
        writer.writeheader()
        writer.writerow({"tick": 0, "t_ns": 1, "dt_ns": 10, "quality_flags": 0, "cpu_load_ppm": 42, "probe_acc": 123, "note": "idle"})
        writer.writerow({"tick": 1, "t_ns": 2, "dt_ns": 20, "quality_flags": 0, "cpu_load_ppm": 44, "probe_acc": 456, "note": "idle"})

    columns = discover_numeric_preview_columns(scene_dir / "preview.csv")
    features = summarize_preview(scene_dir / "preview.csv")

    assert columns == ["cpu_load_ppm", "dt_ns"]
    assert features["cpu_load_ppm_median"] == 43
    assert features["cpu_load_ppm_max"] == 44
    assert features["cpu_load_ppm_variance"] == 1
    assert features["cpu_load_ppm_slope"] == 2
    assert "note_median" not in features
    assert "probe_acc_median" not in features


def test_classifier_uses_extra_numeric_channels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_complete_scene(root, "scene_000001", "baseline_idle", [100, 100, 100], cpu_base=10)
    _write_complete_scene(root, "scene_000002", "cpu_load_no_person", [100, 100, 100], cpu_base=900)

    model = train_project_classifier(DEFAULT_PROJECT)

    assert "cpu_load_ppm" in model.detector_baseline
    assert "cpu_load_ppm_median" in model.label_profiles["cpu_load_no_person"]
    assert "cpu_load_ppm_slope" in model.label_profiles["cpu_load_no_person"]
    assert "cpu_load_ppm" in model.feature_manifest["channels"]
    assert "cpu_load_ppm_variance" in model.feature_manifest["features"]


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
    assert report["answers"]["idle_vs_activity"]["answer"] == "yes"
    assert report["answers"]["useful_signal"]["channel"] in {"cpu_load_ppm", "sleep_drift_ns"}
    assert "best_feature" in report["channel_usefulness_ranking"][0]
    assert "label_distance_matrix" in report


def test_evaluate_scenes_cli_writes_custom_out(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_complete_scene(root, "scene_000001", "baseline_idle", [100, 100, 100], cpu_base=10)
    _write_complete_scene(root, "scene_000002", "baseline_idle", [110, 110, 110], cpu_base=11)
    _write_complete_scene(root, "scene_000003", "typing_burst", [5000, 5100, 5200], cpu_base=100)
    _write_complete_scene(root, "scene_000004", "typing_burst", [5300, 5400, 5500], cpu_base=110)
    out = tmp_path / "reports" / "evaluation_report.json"

    main(["evaluate-scenes", DEFAULT_PROJECT, "--out", str(out)])

    printed = capsys.readouterr().out
    report = json.loads(out.read_text(encoding="utf-8"))
    assert out.exists()
    assert str(out) in printed
    assert "Research answers:" in printed
    assert report["answers"]["idle_vs_activity"]["detail"]


def test_extract_features_and_rank_channels_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_complete_scene(root, "scene_000001", "baseline_idle", [100, 100, 100], cpu_base=10)
    _write_complete_scene(root, "scene_000002", "baseline_idle", [110, 110, 110], cpu_base=11)
    _write_complete_scene(root, "scene_000003", "typing_burst", [5000, 5100, 5200], cpu_base=100)
    out = tmp_path / "features" / "features.json"

    main(["extract-features", DEFAULT_PROJECT, "--out", str(out)])
    extract_out = capsys.readouterr().out
    feature_report = json.loads(out.read_text(encoding="utf-8"))
    assert str(out) in extract_out
    assert feature_report["feature_manifest"]["feature_stats"] == ["median", "mad", "p95", "max", "variance", "slope"]
    assert "cpu_load_ppm_slope" in feature_report["feature_manifest"]["features"]
    assert "probe_acc" in feature_report["feature_manifest"]["ignored_columns"]

    main(["rank-channels", DEFAULT_PROJECT, "--limit", "2"])
    rank_out = capsys.readouterr().out
    assert "Channel ranking: base" in rank_out
    assert "best feature" in rank_out


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

    main(["inspect-scene", DEFAULT_PROJECT, "scene_000001"])
    inspect_scene_out = capsys.readouterr().out
    assert "scene_000001" in inspect_scene_out
    assert "frames=3" in inspect_scene_out

    main(["classify-scene", DEFAULT_PROJECT, "scene_000001"])
    classify_out = capsys.readouterr().out
    assert '"label": "baseline_idle"' in classify_out

    main(["inspect-frame", DEFAULT_PROJECT, "scene_000001", "--tick", "1"])
    inspect_out = capsys.readouterr().out
    assert '"sequence": 1' in inspect_out
    assert '"preview"' in inspect_out

    main(["replay-scene", DEFAULT_PROJECT, "scene_000001", "--limit", "1"])
    replay_out = capsys.readouterr().out
    assert "Replay scene_000001" in replay_out
    assert "classifier=baseline_idle" in replay_out

    out = tmp_path / "debug" / "scene.json"
    main(["export-scene-json", DEFAULT_PROJECT, "scene_000001", "--out", str(out)])
    export_out = capsys.readouterr().out
    exported = json.loads(out.read_text(encoding="utf-8"))
    assert "Exported scene_000001" in export_out
    assert exported["format"] == "dsense-scene-debug-v1"
    assert len(exported["frames"]) == 3

    main(["classify-scene", str(scene_dir), "--project", DEFAULT_PROJECT])
    path_out = capsys.readouterr().out
    assert '"label": "baseline_idle"' in path_out


def test_view_scene_and_export_trace_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_complete_scene(root, "scene_000001", "baseline_idle", [100, 120, 90], cpu_base=10)
    trace_out = tmp_path / "trace" / "scene_trace.json"
    html_out = tmp_path / "trace" / "viewer.html"

    main(["export-trace", DEFAULT_PROJECT, "scene_000001", "--out", str(trace_out)])
    export_out = capsys.readouterr().out
    trace = json.loads(trace_out.read_text(encoding="utf-8"))

    assert "Exported trace for scene_000001" in export_out
    assert trace["format"] == "dsense-trace-v1"
    assert {track["name"] for track in trace["tracks"]} >= {"dt_ns", "sleep_drift_ns", "process_ns_estimate", "cpu_load_ppm"}
    assert [event["name"] for event in trace["events"]] == ["scene_start", "action_start", "action_end", "scene_end"]

    main(["view-scene", DEFAULT_PROJECT, "scene_000001", "--out", str(html_out), "--no-open"])
    view_out = capsys.readouterr().out
    html = html_out.read_text(encoding="utf-8")

    assert str(html_out) in view_out
    assert "dSense Scene Viewer" in html
    assert "trace-data" in html
    assert "cpu_load_ppm" in html
