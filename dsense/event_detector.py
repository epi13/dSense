from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median


@dataclass
class DetectorState:
    score: float = 0.0
    channel: str = "warming_up"
    status: str = "warming_up"
    threshold: float = 6.0
    samples: int = 0


@dataclass
class HeuristicEventDetector:
    tick_hz: int
    learned_baseline: dict[str, dict[str, float]] | None = None
    threshold: float = 6.0
    cooldown_ms: int = 900
    warmup_samples: int | None = None
    window_size: int = 80
    _windows: dict[str, deque[float]] = field(default_factory=dict)
    _last_event_ms: int = -1_000_000
    state: DetectorState = field(default_factory=DetectorState)

    def __post_init__(self) -> None:
        if self.warmup_samples is None:
            self.warmup_samples = max(12, min(80, int(self.tick_hz * 0.75)))
        self._windows = {
            "dt_ns": deque(maxlen=self.window_size),
            "sleep_drift_ns": deque(maxlen=self.window_size),
            "process_ns_estimate": deque(maxlen=self.window_size),
        }
        self.state.threshold = self.threshold

    def update(self, progress: dict[str, object]) -> list[dict[str, object]]:
        values = {
            "dt_ns": float(progress.get("dt_ns", 0) or 0),
            "sleep_drift_ns": abs(float(progress.get("sleep_drift_ns", 0) or 0)),
            "process_ns_estimate": float(progress.get("process_ns_estimate", 0) or 0),
        }
        elapsed_ms = int(progress.get("elapsed_ms", 0) or 0)
        samples = int(progress.get("tick", 0) or 0) + 1

        scores = {}
        for name, value in values.items():
            local_score = self._robust_score(value, self._windows[name])
            learned_score = self._learned_score(name, value)
            scores[name] = max(local_score, learned_score)
        channel, score = max(scores.items(), key=lambda item: item[1])
        self.state = DetectorState(
            score=round(score, 2),
            channel=channel,
            status=self._status(score, samples),
            threshold=self.threshold,
            samples=samples,
        )

        for name, value in values.items():
            self._windows[name].append(value)

        if samples < int(self.warmup_samples or 0):
            return []
        if score < self.threshold:
            return []
        if elapsed_ms - self._last_event_ms < self.cooldown_ms:
            return []

        self._last_event_ms = elapsed_ms
        return [{
            "t_ms": elapsed_ms,
            "event": "heuristic_signal_spike",
            "source": "heuristic",
            "channel": channel,
            "score": round(score, 2),
            "confidence": round(min(1.0, score / (self.threshold * 2.0)), 3),
        }]

    def _robust_score(self, value: float, window: deque[float]) -> float:
        if len(window) < 6:
            return 0.0
        center = median(window)
        deviations = [abs(sample - center) for sample in window]
        mad = median(deviations) or 1.0
        return abs(value - center) / mad

    def _learned_score(self, name: str, value: float) -> float:
        if not self.learned_baseline:
            return 0.0
        profile = self.learned_baseline.get(name)
        if not profile:
            return 0.0
        center = float(profile.get("center", 0.0))
        mad = float(profile.get("mad", 1.0)) or 1.0
        return abs(value - center) / mad

    def _status(self, score: float, samples: int) -> str:
        if samples < int(self.warmup_samples or 0):
            return "warming_up"
        if score >= self.threshold:
            return "event"
        if score >= self.threshold * 0.6:
            return "watching"
        return "quiet"
