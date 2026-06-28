from __future__ import annotations

from collections.abc import Callable

from .manifest import allocate_scene_id, project_path
from .recorder import ProgressCallback, record_scene
from .tui_state import CaptureConfig


class CaptureSessionController:
    def __init__(self, config: CaptureConfig):
        self.config = config

    def allocate_scene(self) -> tuple[str, object]:
        scene_id = allocate_scene_id(self.config.project_name)
        return scene_id, project_path(self.config.project_name) / "scenes" / scene_id

    def record(
        self,
        *,
        scene_id: str | None = None,
        label: str | None = None,
        mode: str = "record",
        progress_callback: ProgressCallback | None = None,
        on_scene: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        scene_id = scene_id or allocate_scene_id(self.config.project_name)
        scene_dir = project_path(self.config.project_name) / "scenes" / scene_id
        scene = record_scene(
            scene_dir,
            scene_id,
            label or self.config.label,
            self.config.duration,
            self.config.tick_hz,
            self.config.pre_roll,
            self.config.action,
            self.config.post_roll,
            self.config.notes,
            mode=mode,
            progress_callback=progress_callback,
            channel_groups=self.config.channel_groups,
        )
        if on_scene is not None:
            on_scene(scene)
        return scene
