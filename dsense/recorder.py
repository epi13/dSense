from __future__ import annotations

import csv, hashlib, json, time
from collections.abc import Callable
from pathlib import Path
from .channels import default_channels
from .channels.sleep_jitter import SleepJitterChannel
from .frame import build_frame, FRAME_SIZE
from .quality import summarize_frames
from .utils.files import ensure_dir, write_json
from .utils.timebase import monotonic_ns, utc_now_iso


ProgressCallback = Callable[[dict[str, object]], list[dict[str, object]] | None]


def record_scene(scene_dir: Path, scene_id: str, label: str, duration: float, tick_hz: int = 100,
                 pre_roll: float = 0.0, action: float | None = None, post_roll: float = 0.0,
                 notes: str = "", mode: str = "record",
                 progress_callback: ProgressCallback | None = None,
                 channel_groups: list[str] | tuple[str, ...] | None = None) -> dict[str, object]:
    ensure_dir(scene_dir)
    interval_ns = int(1_000_000_000 / tick_hz)
    expected = max(1, int(round(duration * tick_hz)))
    selected_groups = list(channel_groups or ("portable",))
    channels = default_channels(selected_groups)
    for ch in channels:
        ch.start()
    rows: list[dict[str, int]] = []
    preview_fields = {"tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags"}
    user_events: list[dict[str, object]] = []
    frames_path = scene_dir / "frames.ds64"
    start_ns = monotonic_ns()
    next_target = start_ns
    with frames_path.open("wb") as fh:
        for tick in range(expected):
            next_target = start_ns + tick * interval_ns
            sleep_s = (next_target - monotonic_ns()) / 1_000_000_000
            if sleep_s > 0:
                time.sleep(sleep_s)
            now = monotonic_ns()
            availability = 0
            quality = 0
            vals = {"dt_ns": 0, "sleep_drift_ns": 0, "process_ns_estimate": 0}
            for ch in channels:
                if isinstance(ch, SleepJitterChannel):
                    ch.set_target(next_target)
                if ch.available():
                    availability |= 1 << ch.bit
                    sample = ch.sample(tick, now)
                    quality |= (sample.quality_flag & 1) << ch.bit
                    numeric_values = {k: int(v) for k, v in sample.values.items() if isinstance(v, (int, float, bool))}
                    vals.update(numeric_values)
                    preview_fields.update(numeric_values)
            frame = build_frame(tick, now, availability, quality, vals.get("dt_ns", 0), vals.get("sleep_drift_ns", 0), vals.get("process_ns_estimate", 0))
            fh.write(frame)
            row = {"tick": tick, "t_ns": now, "quality_flags": quality}
            row.update({key: vals.get(key, 0) for key in preview_fields if key not in {"tick", "t_ns", "quality_flags"}})
            rows.append(row)
            if progress_callback is not None:
                elapsed_ms = int((now - start_ns) / 1_000_000)
                progress = {
                    "tick": tick,
                    "expected": expected,
                    "elapsed_ms": elapsed_ms,
                    "duration_ms": int(duration * 1000),
                    "availability_mask": availability,
                    "quality_flags": quality,
                    "dt_ns": vals.get("dt_ns", 0),
                    "sleep_drift_ns": vals.get("sleep_drift_ns", 0),
                    "process_ns_estimate": vals.get("process_ns_estimate", 0),
                }
                for event in progress_callback(progress) or []:
                    if "event" not in event:
                        continue
                    marked = dict(event)
                    marked.setdefault("t_ms", elapsed_ms)
                    marked.setdefault("source", "user")
                    user_events.append(marked)
    for ch in channels:
        ch.stop()
    preview_path = scene_dir / "preview.csv"
    with preview_path.open("w", newline="", encoding="utf-8") as f:
        fixed = ["tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags"]
        extra = sorted(field for field in preview_fields if field not in fixed)
        writer = csv.DictWriter(f, fieldnames=fixed + extra)
        writer.writeheader(); writer.writerows(rows)
    action_start_ms = int(pre_roll * 1000)
    action_end_ms = int((pre_roll + (action if action is not None else max(0.0, duration - pre_roll - post_roll))) * 1000)
    events = [
        {"t_ms": 0, "event": "scene_start"},
        {"t_ms": action_start_ms, "event": "action_start"},
        {"t_ms": action_end_ms, "event": "action_end"},
        {"t_ms": int(duration * 1000), "event": "scene_end"},
    ]
    events.extend(user_events)
    events = sorted(events, key=lambda e: (int(e.get("t_ms", 0)), e.get("event") == "scene_end"))
    (scene_dir / "events.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in events), encoding="utf-8")
    (scene_dir / "notes.txt").write_text(notes + ("\n" if notes else ""), encoding="utf-8")
    sha = hashlib.sha256(frames_path.read_bytes()).hexdigest()
    (scene_dir / "checksum.txt").write_text(f"sha256  frames.ds64  {sha}\nframe_size_bytes  {FRAME_SIZE}\n", encoding="utf-8")
    quality_summary = summarize_frames(frames_path, expected, interval_ns).to_dict()
    scene = {"scene_id": scene_id, "label": label, "created_utc": utc_now_iso(), "duration_ms": int(duration * 1000), "tick_hz": tick_hz,
             "frame_size_bytes": FRAME_SIZE, "mode": mode, "machine_state": {}, "pre_roll_ms": int(pre_roll * 1000),
             "action_start_ms": action_start_ms, "action_end_ms": action_end_ms, "post_roll_ms": int(post_roll * 1000),
             "channel_groups": selected_groups,
             "channels": [{"id": c.id, "group": getattr(c, "group", "portable"), "available": c.available(), "bit": c.bit} for c in channels], "quality": quality_summary,
             "accepted": True, "notes": notes, "user_event_count": len(user_events)}
    write_json(scene_dir / "scene.json", scene)
    return scene
