from __future__ import annotations

from dataclasses import dataclass, field

from .event_detector import DetectorState


@dataclass
class CaptureConfig:
    project_name: str
    mode: str = "user"
    scenario_index: int = 0
    record_group: bool = False
    auto_detect: bool = True
    label: str = "user_interaction"
    duration: float = 10.0
    pre_roll: float = 2.0
    action: float = 5.0
    post_roll: float = 3.0
    repeat: int = 1
    tick_hz: int = 100
    notes: str = ""
    workload_id: str | None = None
    startup_baseline_status: str = ""
    channel_groups: list[str] | tuple[str, ...] = ("portable",)
    auto_baseline_policy: str = "auto"
    auto_baseline_duration: float = 5.0
    force_auto_baseline: bool = False
    startup_suite_enabled: bool = True
    startup_suite_target: int = 200
    startup_suite_duration: float = 0.2
    startup_suite_seed: int | None = 42
    startup_suite_linux: bool = True
    startup_intelligence: bool = True
    startup_watchers: bool = True
    startup_orbiters: bool = True
    startup_training: bool = True
    live: bool = True
    start_tab: str = "live"
    fast_start: bool = False
    force_startup_update: bool = False
    startup_mode: str = "balanced"
    startup_cache_policy: str = "auto"
    evaluation_mode: str = "fast"
    workers: int | None = None
    profile_startup: bool = False


@dataclass
class RecordingState:
    events: list[dict[str, object]] = field(default_factory=list)
    live_events: list[dict[str, object]] = field(default_factory=list)
    latest: dict[str, object] = field(default_factory=dict)
    detector_state: DetectorState = field(default_factory=DetectorState)
    last_draw: float = 0.0

    def reset(self) -> None:
        self.events.clear()
        self.live_events.clear()
        self.latest.clear()
        self.detector_state = DetectorState()
        self.last_draw = 0.0


@dataclass
class JobState:
    validation_summary: str = "not run"
    watcher_summary: str = "idle"
    transfer_summary: str = "not exported"
    last_validation_result: object | None = None
    messages: list[str] = field(default_factory=list)


@dataclass
class AppState:
    config: CaptureConfig
    channels: list[dict[str, object]] = field(default_factory=list)
    scenes: list[dict[str, object]] = field(default_factory=list)
    baseline: object | None = None
    classifier: object | None = None
    tab_index: int = 0
    phase_index: int = 0
    scene_index: int = 0
    scene_scroll: int = 0
    jobs: JobState = field(default_factory=JobState)
    recording: RecordingState = field(default_factory=RecordingState)
