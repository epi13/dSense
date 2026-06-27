from __future__ import annotations

import time
from pathlib import Path

from .base import ChannelSample


class DiskLatencyChannel:
    id = "disk_latency"
    name = "Filesystem metadata latency probe"
    rate_hz = 10
    bit = 4

    def __init__(self) -> None:
        self.path = Path.cwd()

    def available(self) -> bool:
        try:
            self.path.stat()
            return True
        except OSError:
            return False

    def start(self) -> None:
        pass

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        start = time.perf_counter_ns()
        self.path.stat()
        elapsed = time.perf_counter_ns() - start
        return ChannelSample(self.id, {"disk_stat_latency_ns": int(elapsed)}, 1 if elapsed > 5_000_000 else 0)

    def stop(self) -> None:
        pass
