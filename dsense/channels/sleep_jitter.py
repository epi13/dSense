from __future__ import annotations

from .base import ChannelSample


class SleepJitterChannel:
    id = "sleep_jitter"
    name = "Sleep wake-up drift"
    rate_hz = 100
    bit = 1

    def __init__(self) -> None:
        self.target_ns: int | None = None

    def set_target(self, target_ns: int) -> None:
        self.target_ns = target_ns

    def available(self) -> bool:
        return True

    def start(self) -> None:
        self.target_ns = None

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        drift = 0 if self.target_ns is None else now_ns - self.target_ns
        return ChannelSample(self.id, {"sleep_drift_ns": int(drift)}, 1 if drift > 5_000_000 else 0)

    def stop(self) -> None:
        pass
