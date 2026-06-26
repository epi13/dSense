from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChannelSample:
    channel_id: str
    values: dict[str, int | float | str | bool]
    quality_flag: int = 0


class DSenseChannel(Protocol):
    id: str
    name: str
    rate_hz: int
    bit: int

    def available(self) -> bool: ...
    def start(self) -> None: ...
    def sample(self, tick: int, now_ns: int) -> ChannelSample: ...
    def stop(self) -> None: ...
