from __future__ import annotations

import os

from .base import ChannelSample


class CPULoadChannel:
    id = "cpu_load"
    name = "Portable CPU load proxy"
    rate_hz = 10
    bit = 3

    def available(self) -> bool:
        return hasattr(os, "getloadavg")

    def start(self) -> None:
        pass

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        load1, _, _ = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        normalized = int((load1 / cpu_count) * 1_000_000)
        return ChannelSample(self.id, {"cpu_load_ppm": normalized}, 1 if normalized > 900_000 else 0)

    def stop(self) -> None:
        pass
