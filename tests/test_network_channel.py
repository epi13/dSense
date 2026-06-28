from __future__ import annotations

from dsense.channels.network_latency import NetworkLatencyChannel


def test_network_port_env_parse_failure_is_unavailable(monkeypatch):
    monkeypatch.setenv("DSENSE_NET_HOST", "127.0.0.1")
    monkeypatch.setenv("DSENSE_NET_PORT", "not-a-port")

    channel = NetworkLatencyChannel()

    assert channel.available() is False
    assert "invalid DSENSE_NET_PORT" in channel.reason
