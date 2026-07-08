from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from .manifest import project_path
from .utils.files import ensure_dir, write_json
from .utils.timebase import utc_now_iso


def startup_profile_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "startup_profile.json"


def resolve_worker_count(workers: int | None = None) -> int:
    if workers is not None:
        return max(1, int(workers))
    env_value = os.environ.get("DSENSE_WORKERS", "").strip()
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    cpu_count = os.cpu_count() or 1
    return min(max(1, cpu_count - 1), 6)


@dataclass
class StartupProfiler:
    project_name: str
    worker_count: int = 1
    scene_count: int = 0
    preview_row_count: int = 0
    cache_policy: str = "auto"
    steps: list[dict[str, object]] = field(default_factory=list)
    started_monotonic: float = field(default_factory=monotonic)

    def record_step(self, name: str, status: str, elapsed_s: float, **summary: object) -> None:
        self.steps.append({
            "name": name,
            "status": status,
            "elapsed_s": round(max(0.0, elapsed_s), 6),
            **{key: value for key, value in summary.items() if value is not None},
        })

    def to_dict(self) -> dict[str, object]:
        slowest = self.slowest_step()
        return {
            "format": "dsense-startup-profile-v1",
            "project_name": self.project_name,
            "created_utc": utc_now_iso(),
            "elapsed_s": round(max(0.0, monotonic() - self.started_monotonic), 6),
            "scene_count": self.scene_count,
            "preview_row_count": self.preview_row_count,
            "worker_count": self.worker_count,
            "cache_policy": self.cache_policy,
            "slowest_step": slowest,
            "steps": self.steps,
        }

    def slowest_step(self) -> dict[str, object]:
        timed = [step for step in self.steps if isinstance(step.get("elapsed_s"), (int, float))]
        if not timed:
            return {}
        return dict(max(timed, key=lambda step: float(step.get("elapsed_s", 0.0) or 0.0)))

    def write(self) -> dict[str, object]:
        profile = self.to_dict()
        path = startup_profile_path(self.project_name)
        ensure_dir(path.parent)
        write_json(path, profile)
        return profile
