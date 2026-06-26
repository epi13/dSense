from __future__ import annotations

import time
from .base import ChannelSample


class ProcessProbeChannel:
    id = "process_probe"
    name = "Tiny deterministic process activity probe"
    rate_hz = 100
    bit = 2

    def available(self) -> bool:
        return True

    def start(self) -> None:
        pass

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        # Crude, portable activity probe: how long does fixed Python work take now?
        start = time.perf_counter_ns()
        acc = 0
        for i in range(128):
            acc = (acc * 33 + i + tick) & 0xFFFFFFFF
        elapsed = time.perf_counter_ns() - start
        return ChannelSample(self.id, {"process_ns_estimate": int(elapsed), "probe_acc": acc})

    def stop(self) -> None:
        pass
