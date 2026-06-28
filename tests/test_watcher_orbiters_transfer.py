from pathlib import Path
import csv

from dsense.manifest import DEFAULT_PROJECT, init_project
from dsense.orbiters import evaluate_project_orbiters, make_orbiter_summary, run_scene_orbiters
from dsense.privacy import build_privacy_report
from dsense.transfer import compare_transfer_bundle, export_transfer_bundle, transfer_bundle_path
from dsense.watcher import label_candidate, run_rolling_watcher, run_watcher_scan
from dsense.utils.files import write_json


def test_orbiter_summary_schema():
    summary = make_orbiter_summary(
        "scene_000001",
        {"status": "normal", "channel": "dt_ns"},
        {"label": "baseline_idle", "confidence": 0.9},
        7,
        0,
        1.2,
    )

    assert summary["scene_id"] == "scene_000001"
    assert summary["schema_version"] == "dsense-orbiter-v1"
    assert "summary" in summary
    assert "confidence_disclaimer" in summary
    assert {"timing", "activity", "drift", "privacy", "transfer"} <= set(summary["orbiter_types"])
    assert summary["local_model_adapters"]["remote_calls"] is False
    assert summary["classifier_prediction"]["label"] == "baseline_idle"


def test_orbiter_run_and_evaluate_compare_actual_label(sample_dataset, monkeypatch):
    monkeypatch.setenv("DSENSE_GEMMA_DISABLE", "1")

    summary = run_scene_orbiters(DEFAULT_PROJECT, "scene_000001")
    evaluation = evaluate_project_orbiters(DEFAULT_PROJECT)

    assert summary["actual_label"] == "baseline_idle"
    assert summary["summary_comparison"]["actual_label"] == "baseline_idle"
    assert "predicted_label" in summary["summary_comparison"]
    assert summary["orbiter_types"]["activity"]["disclaimer"]
    assert evaluation["evaluated"] == 2
    assert "accuracy" in evaluation
    assert all("disclaimer" in row for row in evaluation["summaries"])


def test_transfer_bundle_export_and_compare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_project(DEFAULT_PROJECT)

    bundle = export_transfer_bundle(DEFAULT_PROJECT)
    result = compare_transfer_bundle(DEFAULT_PROJECT, transfer_bundle_path(DEFAULT_PROJECT))

    assert bundle["format"] == "dsense-transfer-v1"
    assert Path(transfer_bundle_path(DEFAULT_PROJECT)).exists()
    assert result["compatibility"] in {"compatible", "degraded"}
    assert result["transfer_risk"] in {"low", "medium", "high"}


def test_privacy_report_and_redacted_transfer(sample_dataset):
    report = build_privacy_report(DEFAULT_PROJECT)
    bundle = export_transfer_bundle(DEFAULT_PROJECT, redact=True)

    assert report["scene_count"] == 2
    assert "typing_burst" in report["sensitive_labels"]
    assert bundle["format"] == "dsense-transfer-v1-redacted"
    assert bundle["project_name"] == "redacted"
    assert bundle["sharing_summary"]["contains_labels"] is False
    assert "created_utc" not in bundle
    assert "label_profiles" not in bundle["classifier"]
    assert "label_counts" not in bundle["classifier"]


def test_watcher_scan_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_project(DEFAULT_PROJECT)

    result = run_watcher_scan(DEFAULT_PROJECT, duration=0.05, tick_hz=10)

    assert result["scene"]["mode"] == "watcher"
    assert Path(result["watcher_events_path"]).exists()
    assert Path(result["orbiter_path"]).exists()


def test_rolling_watcher_saves_triggered_window_and_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_baseline_preview(root)
    rows = []
    for tick in range(30):
        drift = 100
        if tick == 15:
            drift = 5_000
        rows.append({
            "tick": tick,
            "t_ns": tick * 100_000_000,
            "dt_ns": 100_000_000,
            "sleep_drift_ns": drift,
            "process_ns_estimate": 8_000,
            "quality_flags": 0,
            "availability_mask": 7,
            "cpu_load_ppm": 10_000,
        })

    result = run_rolling_watcher(
        DEFAULT_PROJECT,
        pre_seconds=0.3,
        post_seconds=0.2,
        tick_hz=10,
        cooldown_seconds=10,
        sample_rows=rows,
    )

    assert len(result["saved"]) == 1
    saved = result["saved"][0]
    scene = saved["scene"]
    scene_dir = root / "scenes" / scene["scene_id"]
    assert scene["mode"] == "watcher_rolling"
    assert scene["label"] == "watcher_anomaly_candidate"
    assert scene["accepted"] is False
    assert (scene_dir / "frames.ds64").exists()
    assert Path(result["session_path"]).exists()

    labeled = label_candidate(DEFAULT_PROJECT, scene["scene_id"], "door_open_close")

    assert labeled["label"] == "door_open_close"
    assert labeled["accepted"] is True


def _write_baseline_preview(root: Path) -> None:
    scene_dir = root / "scenes" / "scene_000001"
    scene_dir.mkdir()
    write_json(scene_dir / "scene.json", {"scene_id": "scene_000001", "label": "baseline_idle", "accepted": True})
    with (scene_dir / "preview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags"])
        writer.writeheader()
        for tick in range(20):
            writer.writerow({
                "tick": tick,
                "t_ns": tick * 100_000_000,
                "dt_ns": 100_000_000,
                "sleep_drift_ns": 100,
                "process_ns_estimate": 8_000,
                "quality_flags": 0,
            })
