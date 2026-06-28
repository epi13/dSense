from __future__ import annotations

from .base import ChannelSample


class ExperimentalEBPFChannel:
    id = "experimental_ebpf"
    name = "Experimental eBPF telemetry adapter"
    rate_hz = 10
    bit = 11
    group = "experimental"

    def available(self) -> bool:
        return False

    def start(self) -> None:
        pass

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        return ChannelSample(self.id, {})

    def stop(self) -> None:
        pass
