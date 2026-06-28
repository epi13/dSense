from .clock_delta import ClockDeltaChannel
from .cpu_load import CPULoadChannel
from .disk_latency import DiskLatencyChannel
from .ebpf import ExperimentalEBPFChannel
from .linux_proc import LinuxMemoryChannel, LinuxProcSelfChannel, LinuxProcStatChannel
from .linux_thermal import LinuxThermalChannel
from .network_latency import NetworkLatencyChannel
from .power_state import PowerStateChannel
from .sleep_jitter import SleepJitterChannel
from .process_probe import ProcessProbeChannel

CHANNEL_GROUPS = ("portable", "linux", "experimental")


def portable_channels():
    return [
        ClockDeltaChannel(),
        SleepJitterChannel(),
        ProcessProbeChannel(),
        CPULoadChannel(),
        DiskLatencyChannel(),
        NetworkLatencyChannel(),
        PowerStateChannel(),
    ]


def linux_channels():
    return [
        LinuxProcStatChannel(),
        LinuxProcSelfChannel(),
        LinuxMemoryChannel(),
        LinuxThermalChannel(),
    ]


def experimental_channels():
    return [
        ExperimentalEBPFChannel(),
    ]


def default_channels(groups: list[str] | tuple[str, ...] | None = None):
    selected = list(groups or ("portable",))
    channels = []
    if "portable" in selected:
        channels.extend(portable_channels())
    if "linux" in selected:
        channels.extend(linux_channels())
    if "experimental" in selected:
        channels.extend(experimental_channels())
    return channels


def parse_channel_groups(value: str | None) -> list[str]:
    if not value:
        return ["portable"]
    groups = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(groups) - set(CHANNEL_GROUPS))
    if unknown:
        raise ValueError(f"Unknown channel group(s): {', '.join(unknown)}. Valid groups: {', '.join(CHANNEL_GROUPS)}")
    return groups
