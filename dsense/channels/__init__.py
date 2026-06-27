from .clock_delta import ClockDeltaChannel
from .cpu_load import CPULoadChannel
from .disk_latency import DiskLatencyChannel
from .network_latency import NetworkLatencyChannel
from .power_state import PowerStateChannel
from .sleep_jitter import SleepJitterChannel
from .process_probe import ProcessProbeChannel


def default_channels():
    return [
        ClockDeltaChannel(),
        SleepJitterChannel(),
        ProcessProbeChannel(),
        CPULoadChannel(),
        DiskLatencyChannel(),
        NetworkLatencyChannel(),
        PowerStateChannel(),
    ]
