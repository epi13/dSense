from __future__ import annotations

from .base import ChannelSample


class ClockDeltaChannel:
    id = "clock_delta"
    name = "Monotonic clock delta"
    rate_hz = 100
    bit = 0

    def __init__(self) -> None:
        self._last_ns: int | None = None

    def available(self) -> bool:
        return True

    def start(self) -> None:
        self._last_ns = None

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        dt_ns = 0 if self._last_ns is None else now_ns - self._last_ns
        self._last_ns = now_ns
        return ChannelSample(self.id, {"dt_ns": int(dt_ns)})

    def stop(self) -> None:
        pass
