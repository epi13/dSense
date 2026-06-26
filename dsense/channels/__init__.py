from .clock_delta import ClockDeltaChannel
from .sleep_jitter import SleepJitterChannel
from .process_probe import ProcessProbeChannel


def default_channels():
    return [ClockDeltaChannel(), SleepJitterChannel(), ProcessProbeChannel()]
