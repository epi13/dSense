from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ChannelSnapshot:
    id: str
    group: str
    bit: int
    available: bool
    sampled: bool
    stale: bool
    unavailable: bool
    value: float | int | None
    quality_flag: int
    rate_hz: float
    last_sample_tick: int | None
    error_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RecordingSnapshot:
    scene_id: str
    label: str
    tick: int
    expected: int
    elapsed_ms: int
    duration_ms: int
    phase: str
    phase_elapsed_ms: int
    availability_mask: int
    quality_flags: int
    sampled_mask: int
    stale_mask: int
    unavailable_mask: int
    values: dict[str, float | int]
    channels: list[ChannelSnapshot]
    detector: dict[str, object]
    recent_events: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_progress_dict(self) -> dict[str, object]:
        progress = self.to_dict()
        progress.update(self.values)
        progress["channel_sampled_mask"] = self.sampled_mask
        progress["channel_stale_mask"] = self.stale_mask
        progress["channel_unavailable_mask"] = self.unavailable_mask
        return progress


def build_channel_snapshots(
    runtimes: list[Any],
    values: dict[str, float | int],
    sampled_mask: int,
    stale_mask: int,
    unavailable_mask: int,
    quality_flags: int,
) -> list[ChannelSnapshot]:
    snapshots: list[ChannelSnapshot] = []
    for runtime in runtimes:
        channel = runtime.channel
        bit = int(getattr(channel, "bit", 0))
        bit_mask = 1 << bit
        last_values = dict(getattr(runtime, "last_values", {}) or {})
        snapshots.append(
            ChannelSnapshot(
                id=str(getattr(channel, "id", "unknown")),
                group=str(getattr(channel, "group", "portable")),
                bit=bit,
                available=bool(getattr(runtime, "available", False)) and not bool(getattr(runtime, "failed", False)),
                sampled=bool(sampled_mask & bit_mask),
                stale=bool(stale_mask & bit_mask),
                unavailable=bool(unavailable_mask & bit_mask),
                value=_primary_value(last_values, values),
                quality_flag=(int(quality_flags) >> bit) & 1,
                rate_hz=float(getattr(channel, "rate_hz", 0) or 0),
                last_sample_tick=getattr(runtime, "last_sample_tick", None),
                error_count=int(getattr(runtime, "error_count", 0) or 0),
                reason=_runtime_reason(runtime),
            )
        )
    return snapshots


def build_recording_snapshot(
    *,
    scene_id: str,
    label: str,
    tick: int,
    expected: int,
    elapsed_ms: int,
    duration_ms: int,
    pre_roll_ms: int,
    action_end_ms: int,
    availability_mask: int,
    quality_flags: int,
    sampled_mask: int,
    stale_mask: int,
    unavailable_mask: int,
    values: dict[str, float | int],
    runtimes: list[Any],
    detector: dict[str, object] | None = None,
    recent_events: list[dict[str, object]] | None = None,
) -> RecordingSnapshot:
    phase, phase_elapsed_ms = phase_at(elapsed_ms, pre_roll_ms, action_end_ms)
    return RecordingSnapshot(
        scene_id=scene_id,
        label=label,
        tick=tick,
        expected=expected,
        elapsed_ms=elapsed_ms,
        duration_ms=duration_ms,
        phase=phase,
        phase_elapsed_ms=phase_elapsed_ms,
        availability_mask=availability_mask,
        quality_flags=quality_flags,
        sampled_mask=sampled_mask,
        stale_mask=stale_mask,
        unavailable_mask=unavailable_mask,
        values=dict(values),
        channels=build_channel_snapshots(runtimes, values, sampled_mask, stale_mask, unavailable_mask, quality_flags),
        detector=dict(detector or {}),
        recent_events=list(recent_events or []),
    )


def phase_at(elapsed_ms: int, pre_roll_ms: int, action_end_ms: int) -> tuple[str, int]:
    if elapsed_ms < pre_roll_ms:
        return "pre-roll", max(0, elapsed_ms)
    if elapsed_ms <= action_end_ms:
        return "action", max(0, elapsed_ms - pre_roll_ms)
    return "post-roll", max(0, elapsed_ms - action_end_ms)


def _primary_value(last_values: dict[str, object], current_values: dict[str, object]) -> float | int | None:
    numeric = [
        value
        for key, value in last_values.items()
        if key in current_values and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(numeric) == 1:
        return numeric[0]
    return None


def _runtime_reason(runtime: Any) -> str:
    reason = str(getattr(runtime, "reason", "") or "")
    if reason:
        return reason
    if bool(getattr(runtime, "available", False)) and not bool(getattr(runtime, "failed", False)):
        return "ok"
    return "unavailable"
