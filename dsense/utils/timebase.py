from __future__ import annotations

from datetime import datetime, timezone
import time


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def monotonic_ns() -> int:
    return time.perf_counter_ns()
