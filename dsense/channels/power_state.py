from __future__ import annotations

from pathlib import Path

from .base import ChannelSample


class PowerStateChannel:
    id = "power_state"
    name = "Linux power/battery state"
    rate_hz = 1
    bit = 6

    def __init__(self) -> None:
        self.root = Path("/sys/class/power_supply")

    def available(self) -> bool:
        return self.root.exists() and any(self.root.iterdir())

    def start(self) -> None:
        pass

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        online = 0
        capacity = -1
        for supply in self.root.iterdir():
            type_path = supply / "type"
            supply_type = type_path.read_text(encoding="utf-8").strip().lower() if type_path.exists() else ""
            if supply_type in {"mains", "usb"}:
                online_path = supply / "online"
                if online_path.exists() and online_path.read_text(encoding="utf-8").strip() == "1":
                    online = 1
            if supply_type == "battery":
                capacity_path = supply / "capacity"
                if capacity_path.exists():
                    try:
                        capacity = int(capacity_path.read_text(encoding="utf-8").strip())
                    except ValueError:
                        capacity = -1
        quality = 1 if capacity != -1 and capacity < 15 and not online else 0
        return ChannelSample(self.id, {"power_online": online, "battery_percent": capacity}, quality)

    def stop(self) -> None:
        pass
