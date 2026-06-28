from __future__ import annotations

from pathlib import Path

from .base import ChannelSample


class LinuxProcStatChannel:
    id = "linux_proc_stat"
    name = "Linux /proc/stat scheduler counters"
    rate_hz = 10
    bit = 7
    group = "linux"

    def __init__(self) -> None:
        self.path = Path("/proc/stat")

    def available(self) -> bool:
        return self.path.exists() and self.path.is_file()

    def start(self) -> None:
        pass

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        values = {"linux_ctxt_total": 0, "linux_procs_running": 0, "linux_procs_blocked": 0}
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "ctxt" and len(parts) > 1:
                values["linux_ctxt_total"] = int(parts[1])
            elif parts[0] == "procs_running" and len(parts) > 1:
                values["linux_procs_running"] = int(parts[1])
            elif parts[0] == "procs_blocked" and len(parts) > 1:
                values["linux_procs_blocked"] = int(parts[1])
        return ChannelSample(self.id, values)

    def stop(self) -> None:
        pass


class LinuxProcSelfChannel:
    id = "linux_proc_self"
    name = "Linux /proc/self process counters"
    rate_hz = 10
    bit = 8
    group = "linux"

    def __init__(self) -> None:
        self.path = Path("/proc/self/status")

    def available(self) -> bool:
        return self.path.exists() and self.path.is_file()

    def start(self) -> None:
        pass

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        values = {
            "linux_self_vmrss_kb": 0,
            "linux_self_voluntary_ctxt": 0,
            "linux_self_nonvoluntary_ctxt": 0,
        }
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, raw = line.partition(":")
            value = raw.strip().split()[0] if raw.strip() else "0"
            if key == "VmRSS":
                values["linux_self_vmrss_kb"] = _safe_int(value)
            elif key == "voluntary_ctxt_switches":
                values["linux_self_voluntary_ctxt"] = _safe_int(value)
            elif key == "nonvoluntary_ctxt_switches":
                values["linux_self_nonvoluntary_ctxt"] = _safe_int(value)
        return ChannelSample(self.id, values)

    def stop(self) -> None:
        pass


class LinuxMemoryChannel:
    id = "linux_memory"
    name = "Linux /proc/meminfo memory state"
    rate_hz = 2
    bit = 9
    group = "linux"

    def __init__(self) -> None:
        self.path = Path("/proc/meminfo")

    def available(self) -> bool:
        return self.path.exists() and self.path.is_file()

    def start(self) -> None:
        pass

    def sample(self, tick: int, now_ns: int) -> ChannelSample:
        values = {"linux_mem_available_kb": 0, "linux_mem_free_kb": 0}
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, raw = line.partition(":")
            value = raw.strip().split()[0] if raw.strip() else "0"
            if key == "MemAvailable":
                values["linux_mem_available_kb"] = _safe_int(value)
            elif key == "MemFree":
                values["linux_mem_free_kb"] = _safe_int(value)
        return ChannelSample(self.id, values)

    def stop(self) -> None:
        pass


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0
