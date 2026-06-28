import json

from dsense.autotest import validate_scene
from dsense.recorder import record_scene


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
