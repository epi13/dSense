from __future__ import annotations

import csv, hashlib, json, time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from .channels import default_channels
from .channels.base import ChannelSample
from .channels.sleep_jitter import SleepJitterChannel
from .frame import INT32_MAX, INT32_MIN, build_frame, FRAME_SIZE
from .inputs import validate_capture_params
from .quality import summarize_frames
from .telemetry import build_recording_snapshot
from .utils.files import ensure_dir, write_json
from .utils.timebase import monotonic_ns, utc_now_iso


ProgressCallback = Callable[[dict[str, object]], list[dict[str, object]] | None]
RAW_OVERFLOW_QUALITY_MASK = 1 << 31


@dataclass
class ChannelRuntime:
    channel: object
    interval_ticks: int
    available: bool = False
    started: bool = False
    failed: bool = False
    reason: str = ""
    last_values: dict[str, int] = field(default_factory=dict)
    sample_count: int = 0
    stale_count: int = 0
    error_count: int = 0
    last_sample_tick: int | None = None

    @property
    def bit_mask(self) -> int:
        return 1 << int(getattr(self.channel, "bit", 0))

    def metadata(self) -> dict[str, object]:
        channel = self.channel
        return {
            "id": getattr(channel, "id", "unknown"),
            "group": getattr(channel, "group", "portable"),
            "available": self.available and not self.failed,
            "bit": getattr(channel, "bit", 0),
            "rate_hz": getattr(channel, "rate_hz", 0),
            "interval_ticks": self.interval_ticks,
            "sample_count": self.sample_count,
            "stale_count": self.stale_count,
            "error_count": self.error_count,
            "reason": self.reason or ("ok" if self.available else "unavailable"),
        }


def record_scene(scene_dir: Path, scene_id: str, label: str, duration: float, tick_hz: int = 100,
                 pre_roll: float = 0.0, action: float | None = None, post_roll: float = 0.0,
                 notes: str = "", mode: str = "record",
                 progress_callback: ProgressCallback | None = None,
                 channel_groups: list[str] | tuple[str, ...] | None = None) -> dict[str, object]:
    validate_capture_params(duration, tick_hz, pre_roll, action, post_roll)
    ensure_dir(scene_dir)
    interval_ns = int(1_000_000_000 / tick_hz)
    expected = max(1, int(round(duration * tick_hz)))
    selected_groups = list(channel_groups or ("portable",))
    runtimes = _prepare_channel_runtimes(default_channels(selected_groups), tick_hz)
    rows: list[dict[str, int]] = []
    preview_fields = {"tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags", "channel_sampled_mask", "channel_stale_mask", "channel_unavailable_mask"}
    user_events: list[dict[str, object]] = []
    frames_path = scene_dir / "frames.ds64"
    frames_sha = hashlib.sha256()
    start_ns = monotonic_ns()
    next_target = start_ns
    action_start_ms = int(pre_roll * 1000)
    action_end_ms = int((pre_roll + (action if action is not None else max(0.0, duration - pre_roll - post_roll))) * 1000)
    duration_ms = int(duration * 1000)
    try:
        with frames_path.open("wb") as fh:
            for tick in range(expected):
                next_target = start_ns + tick * interval_ns
                sleep_s = (next_target - monotonic_ns()) / 1_000_000_000
                if sleep_s > 0:
                    time.sleep(sleep_s)
                now = monotonic_ns()
                vals, availability, quality, sampled_mask, stale_mask, unavailable_mask = _sample_runtimes(runtimes, tick, now, next_target)
                preview_fields.update(vals)
                if _raw_overflowed(vals):
                    quality |= RAW_OVERFLOW_QUALITY_MASK
                frame = build_frame(tick, now, availability, quality, vals.get("dt_ns", 0), vals.get("sleep_drift_ns", 0), vals.get("process_ns_estimate", 0))
                fh.write(frame)
                frames_sha.update(frame)
                row = {
                    "tick": tick,
                    "t_ns": now,
                    "quality_flags": quality,
                    "channel_sampled_mask": sampled_mask,
                    "channel_stale_mask": stale_mask,
                    "channel_unavailable_mask": unavailable_mask,
                }
                row.update({key: vals.get(key, 0) for key in preview_fields if key not in row and key not in {"tick", "t_ns", "quality_flags"}})
                rows.append(row)
                if progress_callback is not None:
                    elapsed_ms = int((now - start_ns) / 1_000_000)
                    snapshot = build_recording_snapshot(
                        scene_id=scene_id,
                        label=label,
                        tick=tick,
                        expected=expected,
                        elapsed_ms=elapsed_ms,
                        duration_ms=duration_ms,
                        pre_roll_ms=action_start_ms,
                        action_end_ms=action_end_ms,
                        availability_mask=availability,
                        quality_flags=quality,
                        sampled_mask=sampled_mask,
                        stale_mask=stale_mask,
                        unavailable_mask=unavailable_mask,
                        values=vals,
                        runtimes=runtimes,
                        recent_events=user_events[-12:],
                    )
                    progress = snapshot.to_progress_dict()
                    for event in progress_callback(progress) or []:
                        if "event" not in event:
                            continue
                        marked = dict(event)
                        marked.setdefault("t_ms", elapsed_ms)
                        marked.setdefault("source", "user")
                        user_events.append(marked)
    finally:
        _stop_channel_runtimes(runtimes)
    preview_path = scene_dir / "preview.csv"
    with preview_path.open("w", newline="", encoding="utf-8") as f:
        fixed = ["tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags"]
        extra = sorted(field for field in preview_fields if field not in fixed)
        writer = csv.DictWriter(f, fieldnames=fixed + extra)
        writer.writeheader(); writer.writerows(rows)
    events = [
        {"t_ms": 0, "event": "scene_start"},
        {"t_ms": action_start_ms, "event": "action_start"},
        {"t_ms": action_end_ms, "event": "action_end"},
        {"t_ms": duration_ms, "event": "scene_end"},
    ]
    events.extend(user_events)
    events = sorted(events, key=lambda e: (int(e.get("t_ms", 0)), e.get("event") == "scene_end"))
    (scene_dir / "events.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in events), encoding="utf-8")
    (scene_dir / "notes.txt").write_text(notes + ("\n" if notes else ""), encoding="utf-8")
    sha = frames_sha.hexdigest()
    (scene_dir / "checksum.txt").write_text(f"sha256  frames.ds64  {sha}\nframe_size_bytes  {FRAME_SIZE}\n", encoding="utf-8")
    quality_summary = summarize_frames(frames_path, expected, interval_ns).to_dict()
    scene = {"scene_id": scene_id, "label": label, "created_utc": utc_now_iso(), "duration_ms": duration_ms, "tick_hz": tick_hz,
             "frame_size_bytes": FRAME_SIZE, "mode": mode, "machine_state": {}, "pre_roll_ms": int(pre_roll * 1000),
             "action_start_ms": action_start_ms, "action_end_ms": action_end_ms, "post_roll_ms": int(post_roll * 1000),
             "channel_groups": selected_groups,
             "channels": [runtime.metadata() for runtime in runtimes], "quality": quality_summary,
             "scheduler": {"mode": "per-channel-rate", "stale_values_reused": True, "sampled_mask_column": "channel_sampled_mask", "stale_mask_column": "channel_stale_mask", "unavailable_mask_column": "channel_unavailable_mask"},
             "accepted": True, "notes": notes, "user_event_count": len(user_events)}
    write_json(scene_dir / "scene.json", scene)
    return scene


def _prepare_channel_runtimes(channels: list[object], tick_hz: int) -> list[ChannelRuntime]:
    runtimes = []
    for channel in channels:
        rate_hz = max(1.0, float(getattr(channel, "rate_hz", tick_hz) or tick_hz))
        interval_ticks = max(1, int(round(tick_hz / rate_hz)))
        runtime = ChannelRuntime(channel, interval_ticks)
        try:
            runtime.available = bool(channel.available())
        except Exception as exc:
            runtime.available = False
            runtime.failed = True
            runtime.error_count += 1
            runtime.reason = f"available failed: {exc}"
        if runtime.available:
            try:
                channel.start()
                runtime.started = True
            except Exception as exc:
                runtime.available = False
                runtime.failed = True
                runtime.error_count += 1
                runtime.reason = f"start failed: {exc}"
        runtimes.append(runtime)
    return runtimes


def _sample_runtimes(runtimes: list[ChannelRuntime], tick: int, now_ns: int, target_ns: int) -> tuple[dict[str, int], int, int, int, int, int]:
    vals: dict[str, int] = {"dt_ns": 0, "sleep_drift_ns": 0, "process_ns_estimate": 0}
    availability = 0
    quality = 0
    sampled_mask = 0
    stale_mask = 0
    unavailable_mask = 0
    for runtime in runtimes:
        channel = runtime.channel
        bit_mask = runtime.bit_mask
        if not runtime.available or runtime.failed:
            unavailable_mask |= bit_mask
            continue
        availability |= bit_mask
        if isinstance(channel, SleepJitterChannel):
            channel.set_target(target_ns)
        due = tick == 0 or tick % runtime.interval_ticks == 0
        if due:
            try:
                sample = channel.sample(tick, now_ns)
                numeric_values = _numeric_sample_values(sample)
                runtime.last_values = numeric_values
                runtime.last_sample_tick = tick
                runtime.sample_count += 1
                sampled_mask |= bit_mask
                quality |= (sample.quality_flag & 1) << int(getattr(channel, "bit", 0))
                vals.update(numeric_values)
            except Exception as exc:
                runtime.available = False
                runtime.failed = True
                runtime.error_count += 1
                runtime.reason = f"sample failed: {exc}"
                unavailable_mask |= bit_mask
        elif runtime.last_values:
            runtime.stale_count += 1
            stale_mask |= bit_mask
            vals.update(runtime.last_values)
    return vals, availability, quality, sampled_mask, stale_mask, unavailable_mask


def _numeric_sample_values(sample: ChannelSample) -> dict[str, int]:
    return {key: int(value) for key, value in sample.values.items() if isinstance(value, (int, float, bool))}


def _stop_channel_runtimes(runtimes: list[ChannelRuntime]) -> None:
    for runtime in runtimes:
        if not runtime.started:
            continue
        try:
            runtime.channel.stop()
        except Exception as exc:
            runtime.error_count += 1
            runtime.reason = runtime.reason or f"stop failed: {exc}"


def _raw_overflowed(vals: dict[str, int]) -> bool:
    for key in ("dt_ns", "sleep_drift_ns", "process_ns_estimate"):
        value = int(vals.get(key, 0))
        if value < INT32_MIN or value > INT32_MAX:
            return True
    return False
