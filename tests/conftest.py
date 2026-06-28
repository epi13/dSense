from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from dsense.frame import FRAME_SIZE, build_frame
from dsense.utils.files import write_json


@pytest.fixture
def sample_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "datasets" / "base"
    (root / "scenes").mkdir(parents=True)
    (root / "exports").mkdir()
    write_json(root / "manifest.json", {
        "project_name": "base",
        "created_utc": "2026-06-27T00:00:00Z",
        "format": "dsense-scene-v0",
        "next_scene": 3,
    })
    _write_fixture_scene(root, "scene_000001", "baseline_idle", [100, 105, 95, 100], cpu_load_ppm=10_000)
    _write_fixture_scene(root, "scene_000002", "typing_burst", [5_000, 5_100, 4_900, 5_200], cpu_load_ppm=30_000)
    return root


def _write_fixture_scene(root: Path, scene_id: str, label: str, drift_values: list[int], cpu_load_ppm: int) -> Path:
    scene_dir = root / "scenes" / scene_id
    scene_dir.mkdir()
    frame_data = b"".join(
        build_frame(tick, 1_000_000_000 + tick * 100_000_000, 7, 0, 100_000_000, drift, 8_000)
        for tick, drift in enumerate(drift_values)
    )
    (scene_dir / "frames.ds64").write_bytes(frame_data)
    with (scene_dir / "preview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "tick",
            "t_ns",
            "dt_ns",
            "sleep_drift_ns",
            "process_ns_estimate",
            "quality_flags",
            "cpu_load_ppm",
        ])
        writer.writeheader()
        for tick, drift in enumerate(drift_values):
            writer.writerow({
                "tick": tick,
                "t_ns": 1_000_000_000 + tick * 100_000_000,
                "dt_ns": 100_000_000,
                "sleep_drift_ns": drift,
                "process_ns_estimate": 8_000,
                "quality_flags": 0,
                "cpu_load_ppm": cpu_load_ppm + tick,
            })
    duration_ms = len(drift_values) * 100
    events = [
        {"t_ms": 0, "event": "scene_start"},
        {"t_ms": 0, "event": "action_start"},
        {"t_ms": duration_ms, "event": "action_end"},
        {"t_ms": duration_ms, "event": "scene_end"},
    ]
    (scene_dir / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    (scene_dir / "notes.txt").write_text("sample fixture\n", encoding="utf-8")
    checksum = hashlib.sha256(frame_data).hexdigest()
    (scene_dir / "checksum.txt").write_text(f"sha256  frames.ds64  {checksum}\nframe_size_bytes  {FRAME_SIZE}\n", encoding="utf-8")
    write_json(scene_dir / "scene.json", {
        "scene_id": scene_id,
        "label": label,
        "created_utc": "2026-06-27T00:00:00Z",
        "duration_ms": duration_ms,
        "tick_hz": 10,
        "frame_size_bytes": FRAME_SIZE,
        "mode": "fixture",
        "quality": {
            "expected_frames": len(drift_values),
            "actual_frames": len(drift_values),
            "confidence": 1.0,
            "checksum_ok": True,
            "frame_size_valid": True,
            "jitter_ns": 0,
        },
        "accepted": True,
    })
    return scene_dir
