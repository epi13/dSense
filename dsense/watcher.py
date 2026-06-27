from __future__ import annotations

import json
from pathlib import Path

from .baseline import load_project_baseline, score_against_baseline, train_and_save_project_baseline
from .classifier import load_project_classifier, predict_scene, train_and_save_project_classifier
from .event_detector import HeuristicEventDetector
from .manifest import allocate_scene_id, project_path
from .orbiters import append_orbiter_summary, make_orbiter_summary
from .recorder import record_scene
from .utils.files import ensure_dir, write_json
from .utils.timebase import utc_now_iso


def run_watcher_scan(project_name: str, duration: float = 5.0, tick_hz: int = 50) -> dict[str, object]:
    root = project_path(project_name)
    baseline = load_project_baseline(project_name) or train_and_save_project_baseline(project_name)
    classifier = load_project_classifier(project_name) or train_and_save_project_classifier(project_name)
    detector = HeuristicEventDetector(tick_hz, learned_baseline=baseline.channels, threshold=baseline.threshold)
    detected: list[dict[str, object]] = []
    latest: dict[str, object] = {}

    def progress(update: dict[str, object]) -> list[dict[str, object]]:
        latest.clear()
        latest.update(update)
        events = detector.update(update)
        detected.extend(events)
        return events

    scene_id = allocate_scene_id(project_name)
    scene_dir = root / "scenes" / scene_id
    label = "watcher_anomaly_candidate"
    scene = record_scene(
        scene_dir,
        scene_id,
        label,
        duration=duration,
        tick_hz=tick_hz,
        pre_roll=0,
        action=duration,
        post_roll=0,
        notes="TUI watcher scan",
        mode="watcher",
        progress_callback=progress,
    )
    if not detected:
        scene["label"] = "watcher_scan"
        scene["accepted"] = False
        write_json(scene_dir / "scene.json", scene)

    prediction = predict_scene(classifier, scene_dir / "preview.csv")
    baseline_status = score_against_baseline({
        "dt_ns": float(latest.get("dt_ns", 0) or 0),
        "sleep_drift_ns": abs(float(latest.get("sleep_drift_ns", 0) or 0)),
        "process_ns_estimate": float(latest.get("process_ns_estimate", 0) or 0),
    }, baseline)
    anomaly_score = float(max([event.get("score", 0.0) for event in detected] or [baseline_status.get("score", 0.0)]))
    event = {
        "created_utc": utc_now_iso(),
        "scene_id": scene_id,
        "event": "watcher_anomaly_candidate" if detected else "watcher_scan_complete",
        "anomaly_score": anomaly_score,
        "strongest_channel": baseline_status.get("channel", "none"),
        "classifier_prediction": prediction,
        "detected_events": detected,
    }
    watcher_events_path = append_watcher_event(root, event)
    summary = make_orbiter_summary(
        scene_id,
        baseline_status,
        prediction,
        int(latest.get("availability_mask", 0) or 0),
        int(latest.get("quality_flags", 0) or 0),
        anomaly_score,
    )
    orbiter_path = append_orbiter_summary(root, summary)
    return {
        "scene": scene,
        "detected": detected,
        "event": event,
        "watcher_events_path": str(watcher_events_path),
        "orbiter_path": str(orbiter_path),
    }


def append_watcher_event(project_root: Path, event: dict[str, object]) -> Path:
    out_dir = ensure_dir(project_root / "watcher")
    path = out_dir / "events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return path


def read_recent_watcher_events(project_name: str, limit: int = 5) -> list[dict[str, object]]:
    path = project_path(project_name) / "watcher" / "events.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows[-limit:]
