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
        self.port = 443
        self.reason = ""
        raw_port = os.environ.get("DSENSE_NET_PORT", "443")
        try:
            self.port = int(raw_port)
            if not (1 <= self.port <= 65535):
                self.reason = f"invalid DSENSE_NET_PORT: {raw_port}"
        except ValueError:
            self.reason = f"invalid DSENSE_NET_PORT: {raw_port}"

    def available(self) -> bool:
        return bool(self.host) and not self.reason

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
