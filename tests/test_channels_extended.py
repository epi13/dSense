from dsense.channels.cpu_load import CPULoadChannel
from dsense.channels.disk_latency import DiskLatencyChannel
from dsense.channels.network_latency import NetworkLatencyChannel
from dsense.channels import default_channels, parse_channel_groups
from dsense.manifest import scan_channels


def test_portable_channels_degrade_gracefully(monkeypatch):
    monkeypatch.delenv("DSENSE_NET_HOST", raising=False)

    cpu = CPULoadChannel()
    disk = DiskLatencyChannel()
    network = NetworkLatencyChannel()

    assert isinstance(cpu.available(), bool)
    assert isinstance(disk.available(), bool)
    assert network.available() is False


def test_advanced_channel_groups_report_permissions():
    portable = scan_channels()
    advanced = scan_channels(advanced=True)
    advanced_ids = {channel["id"] for channel in advanced}

    assert "linux_proc_stat" not in {channel["id"] for channel in portable}
    assert {"linux_proc_stat", "linux_proc_self", "linux_memory", "linux_thermal", "experimental_ebpf"} <= advanced_ids
    assert all("group" in channel and "permission" in channel for channel in advanced)
    assert any(channel["group"] == "experimental" and channel["available"] is False for channel in advanced)


def test_channel_group_parser_and_selection():
    groups = parse_channel_groups("portable,linux")
    channels = default_channels(groups)
    ids = {channel.id for channel in channels}

    assert groups == ["portable", "linux"]
    assert {"clock_delta", "sleep_jitter", "linux_proc_stat"} <= ids
