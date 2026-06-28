import json

import pytest

from dsense.autotest import validate_scene
from dsense.recorder import record_scene
from dsense.channels.base import ChannelSample


def test_record_scene_progress_callback_can_add_events(tmp_path):
    calls = []

    def progress(update):
        calls.append(update)
        if update["tick"] == 0:
            return [{"event": "test_marker", "detail": "from_callback"}]
        return []

    scene = record_scene(
        tmp_path / "scene_000001",
        "scene_000001",
        "test_interaction",
        duration=0.03,
        tick_hz=10,
        progress_callback=progress,
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "scene_000001" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert calls
    assert scene["user_event_count"] == 1
    assert any(event["event"] == "test_marker" for event in events)
    assert events[-1]["event"] == "scene_end"
    assert not [error for error in validate_scene(tmp_path / "scene_000001").errors if error.field == "events"]


def test_record_scene_persists_channel_groups(tmp_path):
    scene = record_scene(
        tmp_path / "scene_000001",
        "scene_000001",
        "linux_channel_test",
        duration=0.03,
        tick_hz=10,
        channel_groups=["portable", "linux"],
    )

    channel_ids = {channel["id"] for channel in scene["channels"]}

    assert scene["channel_groups"] == ["portable", "linux"]
    assert "clock_delta" in channel_ids
    assert "linux_proc_stat" in channel_ids


def test_record_scene_rejects_invalid_runtime_inputs(tmp_path):
    with pytest.raises(ValueError, match="tick_hz"):
        record_scene(tmp_path / "scene_bad", "scene_bad", "bad", duration=0.1, tick_hz=0)
    with pytest.raises(ValueError, match="duration"):
        record_scene(tmp_path / "scene_bad2", "scene_bad2", "bad", duration=0, tick_hz=10)
    with pytest.raises(ValueError, match="pre_roll"):
        record_scene(tmp_path / "scene_bad3", "scene_bad3", "bad", duration=1, tick_hz=10, pre_roll=-1)


def test_record_scene_respects_channel_rate_schedule(tmp_path, monkeypatch):
    class SlowChannel:
        id = "slow"
        name = "Slow"
        rate_hz = 1
        bit = 8

        def __init__(self):
            self.calls = 0

        def available(self):
            return True

        def start(self):
            return None

        def sample(self, tick, now_ns):
            self.calls += 1
            return ChannelSample(self.id, {"slow_value": tick})

        def stop(self):
            return None

    channel = SlowChannel()
    import dsense.recorder

    monkeypatch.setattr(dsense.recorder, "default_channels", lambda groups=None: [channel])

    scene = record_scene(tmp_path / "scene_000001", "scene_000001", "test", duration=0.2, tick_hz=10)
    preview = (tmp_path / "scene_000001" / "preview.csv").read_text(encoding="utf-8")

    assert channel.calls == 1
    assert scene["channels"][0]["sample_count"] == 1
    assert scene["channels"][0]["stale_count"] == 1
    assert "channel_stale_mask" in preview


def test_record_scene_isolates_channel_failures(tmp_path, monkeypatch):
    class BadSampleChannel:
        id = "bad_sample"
        name = "Bad Sample"
        rate_hz = 10
        bit = 9

        def available(self):
            return True

        def start(self):
            return None

        def sample(self, tick, now_ns):
            raise RuntimeError("boom")

        def stop(self):
            return None

    class BadStopChannel(BadSampleChannel):
        id = "bad_stop"
        bit = 10

        def sample(self, tick, now_ns):
            return ChannelSample(self.id, {"stop_value": tick})

        def stop(self):
            raise RuntimeError("stop boom")

    import dsense.recorder

    monkeypatch.setattr(dsense.recorder, "default_channels", lambda groups=None: [BadSampleChannel(), BadStopChannel()])

    scene = record_scene(tmp_path / "scene_000001", "scene_000001", "test", duration=0.1, tick_hz=10)
    statuses = {channel["id"]: channel for channel in scene["channels"]}

    assert statuses["bad_sample"]["available"] is False
    assert statuses["bad_sample"]["error_count"] >= 1
    assert statuses["bad_stop"]["error_count"] >= 1
    assert validate_scene(tmp_path / "scene_000001").valid


def test_record_scene_isolates_available_and_start_failures(tmp_path, monkeypatch):
    class BadAvailableChannel:
        id = "bad_available"
        name = "Bad Available"
        rate_hz = 10
        bit = 11

        def available(self):
            raise RuntimeError("available boom")

        def start(self):
            raise AssertionError("start should not run")

        def sample(self, tick, now_ns):
            raise AssertionError("sample should not run")

        def stop(self):
            raise AssertionError("stop should not run")

    class BadStartChannel:
        id = "bad_start"
        name = "Bad Start"
        rate_hz = 10
        bit = 12

        def available(self):
            return True

        def start(self):
            raise RuntimeError("start boom")

        def sample(self, tick, now_ns):
            raise AssertionError("sample should not run")

        def stop(self):
            raise AssertionError("stop should not run")

    import dsense.recorder

    monkeypatch.setattr(dsense.recorder, "default_channels", lambda groups=None: [BadAvailableChannel(), BadStartChannel()])

    scene = record_scene(tmp_path / "scene_000001", "scene_000001", "test", duration=0.1, tick_hz=10)
    statuses = {channel["id"]: channel for channel in scene["channels"]}

    assert statuses["bad_available"]["available"] is False
    assert "available failed" in statuses["bad_available"]["reason"]
    assert statuses["bad_start"]["available"] is False
    assert "start failed" in statuses["bad_start"]["reason"]
