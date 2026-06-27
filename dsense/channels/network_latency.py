from __future__ import annotations

import os
import socket
import time

from .base import ChannelSample


class NetworkLatencyChannel:
    id = "network_latency"
    name = "Optional TCP connect latency probe"
    rate_hz = 1
    bit = 5

    def __init__(self) -> None:
        self.host = os.environ.get("DSENSE_NET_HOST", "")
        self.port = int(os.environ.get("DSENSE_NET_PORT", "443"))

    def available(self) -> bool:
        return bool(self.host)

    def start(self) -> None:
        pass

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        if not self.host:
            return ChannelSample(self.id, {"network_latency_ns": 0}, 1)
        start = time.perf_counter_ns()
        quality = 0
        try:
            with socket.create_connection((self.host, self.port), timeout=0.15):
                pass
        except OSError:
            quality = 1
        elapsed = time.perf_counter_ns() - start
        return ChannelSample(self.id, {"network_latency_ns": int(elapsed)}, quality)

    def stop(self) -> None:
        pass
