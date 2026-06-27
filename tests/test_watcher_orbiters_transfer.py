from pathlib import Path

from dsense.manifest import DEFAULT_PROJECT, init_project
from dsense.orbiters import make_orbiter_summary
from dsense.transfer import compare_transfer_bundle, export_transfer_bundle, transfer_bundle_path
from dsense.watcher import run_watcher_scan


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
    assert "summary" in summary
    assert summary["classifier_prediction"]["label"] == "baseline_idle"


def test_transfer_bundle_export_and_compare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_project(DEFAULT_PROJECT)

    bundle = export_transfer_bundle(DEFAULT_PROJECT)
    result = compare_transfer_bundle(DEFAULT_PROJECT, transfer_bundle_path(DEFAULT_PROJECT))

    assert bundle["format"] == "dsense-transfer-v1"
    assert Path(transfer_bundle_path(DEFAULT_PROJECT)).exists()
    assert result["compatibility"] in {"compatible", "degraded"}
    assert result["transfer_risk"] in {"low", "medium", "high"}


def test_watcher_scan_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_project(DEFAULT_PROJECT)

    result = run_watcher_scan(DEFAULT_PROJECT, duration=0.05, tick_hz=10)

    assert result["scene"]["mode"] == "watcher"
    assert Path(result["watcher_events_path"]).exists()
    assert Path(result["orbiter_path"]).exists()
