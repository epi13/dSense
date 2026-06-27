from dsense.channels.cpu_load import CPULoadChannel
from dsense.channels.disk_latency import DiskLatencyChannel
from dsense.channels.network_latency import NetworkLatencyChannel


def test_portable_channels_degrade_gracefully(monkeypatch):
    monkeypatch.delenv("DSENSE_NET_HOST", raising=False)

    cpu = CPULoadChannel()
    disk = DiskLatencyChannel()
    network = NetworkLatencyChannel()

    assert isinstance(cpu.available(), bool)
    assert isinstance(disk.available(), bool)
    assert network.available() is False
