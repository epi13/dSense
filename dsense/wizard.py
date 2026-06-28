from __future__ import annotations

import time
from .manifest import allocate_scene_id, project_path
from .recorder import record_scene
from .utils.files import write_json


def countdown(seconds: int = 3) -> None:
    print("Prepare scene.")
    for n in range(seconds, 0, -1):
        print(f"Recording starts in {n}...")
        time.sleep(1)


def guided_scene(project: str, label: str, duration: float, pre_roll: float, action: float, post_roll: float,
                 repeat: int = 1, notes: str = "", tick_hz: int = 100, assume_yes: bool = False,
                 channel_groups: list[str] | tuple[str, ...] | None = None) -> list[dict[str, object]]:
    results = []
    for r in range(1, repeat + 1):
        print(f"Scene '{label}' repeat {r}/{repeat}")
        countdown()
        print(f"Pre-roll ({pre_roll:g}s): hold control state.")
        print("Action start/end markers will be written automatically.")
        scene_id = allocate_scene_id(project)
        scene_dir = project_path(project) / "scenes" / scene_id
        scene = record_scene(scene_dir, scene_id, label, duration, tick_hz, pre_roll, action, post_roll, notes, channel_groups=channel_groups)
        print("Done.")
        print(f"Quality result: confidence={scene['quality']['confidence']} checksum_ok={scene['quality']['checksum_ok']}")
        keep = "k" if assume_yes else input("Keep, retake, or discard? [k/r/d]: ").strip().lower()[:1] or "k"
        if keep == "d":
            scene["accepted"] = False
        elif keep == "r":
            print("Retake requested; current take marked unaccepted.")
            scene["accepted"] = False
        write_json(scene_dir / "scene.json", scene)
        results.append(scene)
    return results
