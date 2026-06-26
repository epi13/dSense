from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

MAGIC = b"DS01"
FRAME_SIZE = 64
HEADER = struct.Struct("<4sIQII")
RAW = struct.Struct("<iiiI")  # dt_ns, sleep_drift_ns, process_ns_estimate, reserved


@dataclass(frozen=True)
class ParsedFrame:
    magic: bytes
    sequence: int
    t_ns: int
    availability_mask: int
    quality_mask: int
    dt_ns: int
    sleep_drift_ns: int
    process_ns_estimate: int
    raw_reserved: int
    mix: bytes
    digest: bytes


def _digest(data: bytes, size: int = 8) -> bytes:
    return hashlib.blake2s(data, digest_size=size).digest()


def build_frame(sequence: int, t_ns: int, availability_mask: int, quality_mask: int,
                dt_ns: int = 0, sleep_drift_ns: int = 0, process_ns_estimate: int = 0,
                raw_reserved: int = 0) -> bytes:
    raw = RAW.pack(int(dt_ns), int(sleep_drift_ns), int(process_ns_estimate), int(raw_reserved))
    header = HEADER.pack(MAGIC, sequence & 0xFFFFFFFF, t_ns & 0xFFFFFFFFFFFFFFFF,
                         availability_mask & 0xFFFFFFFF, quality_mask & 0xFFFFFFFF)
    mix = _digest(header + raw, 16)
    body = header + raw + mix
    return body + _digest(body, 8)


def parse_frame(frame: bytes) -> ParsedFrame:
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"frame must be {FRAME_SIZE} bytes, got {len(frame)}")
    magic, sequence, t_ns, availability_mask, quality_mask = HEADER.unpack(frame[:24])
    if magic != MAGIC:
        raise ValueError(f"bad frame magic {magic!r}")
    dt_ns, sleep_drift_ns, process_ns_estimate, raw_reserved = RAW.unpack(frame[24:40])
    return ParsedFrame(magic, sequence, t_ns, availability_mask, quality_mask,
                       dt_ns, sleep_drift_ns, process_ns_estimate, raw_reserved,
                       frame[40:56], frame[56:64])


def verify_frame(frame: bytes) -> bool:
    return len(frame) == FRAME_SIZE and frame[56:64] == _digest(frame[:56], 8)


def frame_to_dict(frame: bytes) -> dict[str, int | str | bool]:
    parsed = parse_frame(frame)
    return {
        "magic": parsed.magic.decode("ascii"),
        "sequence": parsed.sequence,
        "t_ns": parsed.t_ns,
        "availability_mask": parsed.availability_mask,
        "quality_mask": parsed.quality_mask,
        "dt_ns": parsed.dt_ns,
        "sleep_drift_ns": parsed.sleep_drift_ns,
        "process_ns_estimate": parsed.process_ns_estimate,
        "checksum_ok": verify_frame(frame),
    }
