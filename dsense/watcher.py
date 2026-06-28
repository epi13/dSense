from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path

from .baseline import load_project_baseline, score_against_baseline, train_and_save_project_baseline
from .channels import default_channels
from .classifier import load_project_classifier, predict_scene, train_and_save_project_classifier
from .event_detector import HeuristicEventDetector
from .frame import FRAME_SIZE, build_frame
from .inputs import validate_capture_params
from .manifest import allocate_scene_id, project_path
from .orbiters import append_orbiter_summary, make_orbiter_summary
from .quality import summarize_frames
from .recorder import RAW_OVERFLOW_QUALITY_MASK, _prepare_channel_runtimes, _raw_overflowed, _sample_runtimes, _stop_channel_runtimes, record_scene
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import monotonic_ns, utc_now_iso


def run_watcher_scan(
    project_name: str,
    duration: float = 5.0,
    tick_hz: int = 50,
    channel_groups: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    validate_capture_params(duration, tick_hz)
    selected_groups = list(channel_groups or ("portable",))
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
        channel_groups=selected_groups,
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
        "channel_groups": selected_groups,
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


def run_rolling_watcher(
    project_name: str,
    pre_seconds: float = 5.0,
    post_seconds: float = 10.0,
    tick_hz: int = 50,
    cooldown_seconds: float = 30.0,
    duration: float = 0.0,
    sample_rows: Iterable[dict[str, object]] | None = None,
    prompt_label: bool = False,
    channel_groups: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    validate_capture_params(max(duration, 0.01) if duration else 0.01, tick_hz)
    if pre_seconds < 0 or post_seconds < 0 or cooldown_seconds < 0:
        raise ValueError("pre_seconds, post_seconds, and cooldown_seconds must be >= 0")
    selected_groups = list(channel_groups or ("portable",))
    root = project_path(project_name)
    baseline = load_project_baseline(project_name) or train_and_save_project_baseline(project_name)
    classifier = load_project_classifier(project_name) or train_and_save_project_classifier(project_name)
    detector = HeuristicEventDetector(tick_hz, learned_baseline=baseline.channels, threshold=baseline.threshold)
    pre_frames = max(1, int(round(pre_seconds * tick_hz)))
    post_frames = max(1, int(round(post_seconds * tick_hz)))
    cooldown_frames = max(0, int(round(cooldown_seconds * tick_hz)))
    interval_ns = int(1_000_000_000 / tick_hz)
    buffer: deque[dict[str, object]] = deque(maxlen=pre_frames)
    saved: list[dict[str, object]] = []
    session = {
        "created_utc": utc_now_iso(),
        "project_name": project_name,
        "mode": "rolling",
        "pre_seconds": pre_seconds,
        "post_seconds": post_seconds,
        "tick_hz": tick_hz,
        "cooldown_seconds": cooldown_seconds,
        "channel_groups": selected_groups,
        "windows": [],
    }
    cooldown_until_tick = -1
    max_ticks = int(round(duration * tick_hz)) if duration > 0 else None

    samples = sample_rows if sample_rows is not None else _live_samples(tick_hz, interval_ns, selected_groups)
    iterator = iter(samples)
    tick = 0
    while max_ticks is None or tick < max_ticks:
        try:
            sample = _normalize_sample(next(iterator), tick, interval_ns)
        except StopIteration:
            break
        tick = int(sample["tick"])
        buffer.append(sample)
        events = detector.update(_progress_from_sample(sample, tick_hz))
        if events and tick >= cooldown_until_tick:
            captured = list(buffer)
            for _ in range(post_frames):
                try:
                    tick += 1
                    post_sample = _normalize_sample(next(iterator), tick, interval_ns)
                except StopIteration:
                    break
                captured.append(post_sample)
                buffer.append(post_sample)
                detector.update(_progress_from_sample(post_sample, tick_hz))
            scene_id = allocate_scene_id(project_name)
            scene_dir = root / "scenes" / scene_id
            label = "watcher_anomaly_candidate"
            if prompt_label:
                response = input(f"Label anomaly {scene_id} (blank keeps {label}): ").strip()
                if response:
                    label = response
            scene = _write_rolling_scene(
                scene_dir,
                scene_id,
                label,
                captured,
                tick_hz,
                pre_seconds,
                post_seconds,
                events,
                accepted=label != "watcher_anomaly_candidate",
                channel_groups=selected_groups,
            )
            prediction = predict_scene(classifier, scene_dir / "preview.csv")
            baseline_status = score_against_baseline(_core_values(captured[-1] if captured else sample), baseline)
            anomaly_score = float(max(event.get("score", 0.0) for event in events))
            event = {
                "created_utc": utc_now_iso(),
                "scene_id": scene_id,
                "event": "watcher_anomaly_window",
                "anomaly_score": anomaly_score,
                "strongest_channel": baseline_status.get("channel", "none"),
                "classifier_prediction": prediction,
                "detected_events": events,
                "pre_seconds": pre_seconds,
                "post_seconds": post_seconds,
                "channel_groups": selected_groups,
            }
            watcher_events_path = append_watcher_event(root, event)
            session["windows"].append(event)
            saved.append({"scene": scene, "event": event, "watcher_events_path": str(watcher_events_path)})
            cooldown_until_tick = tick + cooldown_frames
        tick += 1

    session_path = append_watcher_session(root, session)
    return {"saved": saved, "session": session, "session_path": str(session_path)}


def label_candidate(project_name: str, scene_id: str, label: str, notes: str = "") -> dict[str, object]:
    scene_dir = project_path(project_name) / "scenes" / scene_id
    scene_path = scene_dir / "scene.json"
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing scene.json for {scene_id}")
    scene = read_json(scene_path)
    old_label = str(scene.get("label", "unknown"))
    scene["label"] = label
    scene["accepted"] = True
    scene["labeled_utc"] = utc_now_iso()
    scene["previous_label"] = old_label
    if notes:
        scene["notes"] = notes
        (scene_dir / "notes.txt").write_text(notes + "\n", encoding="utf-8")
    write_json(scene_path, scene)
    event = {
        "created_utc": utc_now_iso(),
        "scene_id": scene_id,
        "event": "watcher_candidate_labeled",
        "old_label": old_label,
        "label": label,
    }
    append_watcher_event(project_path(project_name), event)
    return scene


def append_watcher_event(project_root: Path, event: dict[str, object]) -> Path:
    out_dir = ensure_dir(project_root / "watcher")
    path = out_dir / "events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return path


def append_watcher_session(project_root: Path, session: dict[str, object]) -> Path:
    out_dir = ensure_dir(project_root / "watcher")
    path = out_dir / "sessions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(session, sort_keys=True) + "\n")
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


def _live_samples(
    tick_hz: int,
    interval_ns: int,
    channel_groups: list[str] | tuple[str, ...] | None = None,
) -> Iterable[dict[str, object]]:
    channels = _prepare_channel_runtimes(default_channels(channel_groups), tick_hz)
    start_ns = monotonic_ns()
    tick = 0
    try:
        while True:
            target_ns = start_ns + tick * interval_ns
            sleep_s = (target_ns - monotonic_ns()) / 1_000_000_000
            if sleep_s > 0:
                time.sleep(sleep_s)
            now = monotonic_ns()
            values, availability, quality, sampled_mask, stale_mask, unavailable_mask = _sample_runtimes(channels, tick, now, target_ns)
            yield {
                "tick": tick,
                "t_ns": now,
                "availability_mask": availability,
                "quality_flags": quality,
                "channel_sampled_mask": sampled_mask,
                "channel_stale_mask": stale_mask,
                "channel_unavailable_mask": unavailable_mask,
                **values,
            }
            tick += 1
    finally:
        _stop_channel_runtimes(channels)


def _normalize_sample(sample: dict[str, object], tick: int, interval_ns: int) -> dict[str, object]:
    normalized = dict(sample)
    normalized.setdefault("tick", tick)
    normalized.setdefault("t_ns", tick * interval_ns)
    normalized.setdefault("availability_mask", 0)
    normalized.setdefault("quality_flags", 0)
    normalized.setdefault("dt_ns", interval_ns if tick else 0)
    normalized.setdefault("sleep_drift_ns", 0)
    normalized.setdefault("process_ns_estimate", 0)
    return normalized


def _progress_from_sample(sample: dict[str, object], tick_hz: int) -> dict[str, object]:
    tick = int(sample.get("tick", 0) or 0)
    progress = {
        "tick": tick,
        "elapsed_ms": int(tick * 1000 / max(tick_hz, 1)),
        "dt_ns": sample.get("dt_ns", 0),
        "sleep_drift_ns": sample.get("sleep_drift_ns", 0),
        "process_ns_estimate": sample.get("process_ns_estimate", 0),
        "availability_mask": sample.get("availability_mask", 0),
        "quality_flags": sample.get("quality_flags", 0),
    }
    values = {
        key: value
        for key, value in sample.items()
        if key not in progress and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    progress["values"] = values
    progress.update(values)
    return progress


def _write_rolling_scene(
    scene_dir: Path,
    scene_id: str,
    label: str,
    samples: list[dict[str, object]],
    tick_hz: int,
    pre_seconds: float,
    post_seconds: float,
    detected_events: list[dict[str, object]],
    accepted: bool,
    channel_groups: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    ensure_dir(scene_dir)
    duration_ms = int(len(samples) * 1000 / max(tick_hz, 1))
    interval_ns = int(1_000_000_000 / tick_hz)
    frames_path = scene_dir / "frames.ds64"
    with frames_path.open("wb") as handle:
        for idx, sample in enumerate(samples):
            raw_values = {
                "dt_ns": int(float(sample.get("dt_ns", 0) or 0)),
                "sleep_drift_ns": int(float(sample.get("sleep_drift_ns", 0) or 0)),
                "process_ns_estimate": int(float(sample.get("process_ns_estimate", 0) or 0)),
            }
            quality_flags = int(sample.get("quality_flags", 0) or 0)
            if _raw_overflowed(raw_values):
                quality_flags |= RAW_OVERFLOW_QUALITY_MASK
            handle.write(build_frame(
                idx,
                int(sample.get("t_ns", idx * interval_ns) or 0),
                int(sample.get("availability_mask", 0) or 0),
                quality_flags,
                raw_values["dt_ns"],
                raw_values["sleep_drift_ns"],
                raw_values["process_ns_estimate"],
            ))
    _write_preview(scene_dir / "preview.csv", samples)
    shifted_events = [
        {"t_ms": 0, "event": "scene_start"},
        {"t_ms": int(pre_seconds * 1000), "event": "action_start"},
        {"t_ms": min(duration_ms, int((pre_seconds + post_seconds) * 1000)), "event": "action_end"},
        {"t_ms": duration_ms, "event": "scene_end"},
    ]
    shifted_events.extend({**event, "t_ms": min(duration_ms, max(0, int(event.get("t_ms", int(pre_seconds * 1000)) or 0)))} for event in detected_events)
    (scene_dir / "events.jsonl").write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in sorted(shifted_events, key=lambda item: int(item.get("t_ms", 0) or 0))), encoding="utf-8")
    (scene_dir / "notes.txt").write_text("Rolling watcher anomaly window\n", encoding="utf-8")
    sha = hashlib.sha256(frames_path.read_bytes()).hexdigest()
    (scene_dir / "checksum.txt").write_text(f"sha256  frames.ds64  {sha}\nframe_size_bytes  {FRAME_SIZE}\n", encoding="utf-8")
    quality = summarize_frames(frames_path, len(samples), interval_ns).to_dict()
    scene = {
        "scene_id": scene_id,
        "label": label,
        "created_utc": utc_now_iso(),
        "duration_ms": duration_ms,
        "tick_hz": tick_hz,
        "frame_size_bytes": FRAME_SIZE,
        "mode": "watcher_rolling",
        "pre_roll_ms": int(pre_seconds * 1000),
        "action_start_ms": int(pre_seconds * 1000),
        "action_end_ms": min(duration_ms, int((pre_seconds + post_seconds) * 1000)),
        "post_roll_ms": int(post_seconds * 1000),
        "channel_groups": list(channel_groups or ("portable",)),
        "quality": quality,
        "accepted": accepted,
        "notes": "Rolling watcher anomaly window",
        "user_event_count": len(detected_events),
    }
    write_json(scene_dir / "scene.json", scene)
    return scene


def _write_preview(path: Path, samples: list[dict[str, object]]) -> None:
    fixed = ["tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags"]
    fields = set(fixed)
    for sample in samples:
        fields.update(key for key, value in sample.items() if isinstance(value, (int, float, bool)))
    extra = sorted(field for field in fields if field not in fixed and field != "availability_mask")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fixed + extra)
        writer.writeheader()
        for idx, sample in enumerate(samples):
            row = {field: sample.get(field, 0) for field in fixed + extra}
            row["tick"] = idx
            writer.writerow(row)


def _core_values(sample: dict[str, object]) -> dict[str, float]:
    return {
        "dt_ns": float(sample.get("dt_ns", 0) or 0),
        "sleep_drift_ns": abs(float(sample.get("sleep_drift_ns", 0) or 0)),
        "process_ns_estimate": float(sample.get("process_ns_estimate", 0) or 0),
    }
