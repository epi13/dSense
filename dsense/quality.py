from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from .frame import FRAME_SIZE, verify_frame, parse_frame


@dataclass(frozen=True)
class QualitySummary:
    expected_frames: int
    actual_frames: int
    frame_size_valid: bool
    dropped_or_late_estimate: int
    average_dt_ns: float
    min_dt_ns: int
    max_dt_ns: int
    jitter_ns: float
    checksum_ok: bool
    channel_availability_mask: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_frames(frames_path: Path, expected_frames: int, target_interval_ns: int) -> QualitySummary:
    data_len = frames_path.stat().st_size if frames_path.exists() else 0
    frame_size_valid = data_len % FRAME_SIZE == 0
    frames = _read_frames(frames_path) if frame_size_valid else []
    checks = [verify_frame(f) for f in frames]
    parsed = [parse_frame(f) for f, ok in zip(frames, checks) if ok]
    dts = [p.dt_ns for p in parsed if p.dt_ns > 0]
    avg = sum(dts) / len(dts) if dts else 0.0
    mn = min(dts) if dts else 0
    mx = max(dts) if dts else 0
    jitter = (sum(abs(dt - target_interval_ns) for dt in dts) / len(dts)) if dts else 0.0
    late = sum(1 for dt in dts if dt > target_interval_ns * 1.5)
    dropped = abs(expected_frames - len(frames)) + late
    availability = 0
    for p in parsed:
        availability |= p.availability_mask
    frame_score = 1.0 if expected_frames == 0 else max(0.0, 1.0 - abs(expected_frames - len(frames)) / expected_frames)
    jitter_score = max(0.0, 1.0 - (jitter / max(target_interval_ns, 1)))
    checksum_score = 1.0 if all(checks) and frame_size_valid else 0.0
    confidence = round((frame_score * 0.45 + jitter_score * 0.35 + checksum_score * 0.20), 3)
    return QualitySummary(expected_frames, len(frames), frame_size_valid, dropped, avg, mn, mx, jitter,
                          all(checks) and frame_size_valid, availability, confidence)


def _read_frames(frames_path: Path) -> list[bytes]:
    frames: list[bytes] = []
    if not frames_path.exists():
        return frames
    with frames_path.open("rb") as handle:
        while True:
            frame = handle.read(FRAME_SIZE)
            if not frame:
                break
            frames.append(frame)
    return frames
