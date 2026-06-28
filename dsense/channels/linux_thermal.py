from __future__ import annotations

from pathlib import Path

from .base import ChannelSample


class LinuxThermalChannel:
    id = "linux_thermal"
    name = "Linux thermal zone temperature"
    rate_hz = 1
    bit = 10
    group = "linux"

    def __init__(self) -> None:
        self.root = Path("/sys/class/thermal")

    def available(self) -> bool:
        return self._first_temp_path() is not None

    def start(self) -> None:
        pass

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        path = self._first_temp_path()
        temp_millic = 0
        if path is not None:
            try:
                temp_millic = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                temp_millic = 0
        return ChannelSample(self.id, {"linux_cpu_temp_millic": temp_millic}, 1 if temp_millic > 85_000 else 0)

    def stop(self) -> None:
        pass

    def _first_temp_path(self) -> Path | None:
        if not self.root.exists():
            return None
        for zone in sorted(self.root.glob("thermal_zone*/temp")):
            if zone.is_file():
                return zone
        return None
