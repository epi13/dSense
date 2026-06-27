from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .manifest import project_path
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso

BASELINE_CHANNELS = ("dt_ns", "sleep_drift_ns", "process_ns_estimate")


@dataclass(frozen=True)
class BaselineModel:
    project_name: str
    trained_utc: str
    scene_count: int
    threshold: float
    channels: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "project_name": self.project_name,
            "trained_utc": self.trained_utc,
            "scene_count": self.scene_count,
            "threshold": self.threshold,
            "channels": self.channels,
        }


def baseline_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "baseline_model.json"


def train_project_baseline(project_name: str, threshold: float = 6.0) -> BaselineModel:
    root = project_path(project_name)
    values: dict[str, list[float]] = {channel: [] for channel in BASELINE_CHANNELS}
    scene_count = 0
    for scene_path in sorted((root / "scenes").glob("scene_*/scene.json")):
        try:
            scene = read_json(scene_path)
        except (OSError, ValueError):
            continue
        if scene.get("accepted") is False or not str(scene.get("label", "")).startswith("baseline_"):
            continue
        preview = scene_path.parent / "preview.csv"
        if not preview.exists():
            continue
        rows = _read_rows(preview)
        if not rows:
            continue
        scene_count += 1
        for row in rows:
            for channel in BASELINE_CHANNELS:
                values[channel].append(abs(float(row.get(channel, 0.0))))

    channels = {
        channel: _profile(channel_values)
        for channel, channel_values in values.items()
        if channel_values
    }
    return BaselineModel(project_name, utc_now_iso(), scene_count, threshold, channels)


def train_and_save_project_baseline(project_name: str, threshold: float = 6.0) -> BaselineModel:
    model = train_project_baseline(project_name, threshold)
    out = baseline_path(project_name)
    ensure_dir(out.parent)
    write_json(out, model.to_dict())
    return model


def load_project_baseline(project_name: str) -> BaselineModel | None:
    path = baseline_path(project_name)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    return BaselineModel(
        project_name=str(data.get("project_name", project_name)),
        trained_utc=str(data.get("trained_utc", "")),
        scene_count=int(data.get("scene_count", 0)),
        threshold=float(data.get("threshold", 6.0)),
        channels={
            str(channel): {str(k): float(v) for k, v in dict(profile).items()}
            for channel, profile in dict(data.get("channels", {})).items()
        },
    )


def score_against_baseline(values: dict[str, float], model: BaselineModel | None) -> dict[str, object]:
    if model is None or not model.channels:
        return {"score": 0.0, "channel": "none", "status": "untrained", "threshold": 6.0}
    scores = {}
    for channel, value in values.items():
        profile = model.channels.get(channel)
        if not profile:
            continue
        center = float(profile.get("center", 0.0))
        mad = float(profile.get("mad", 1.0)) or 1.0
        scores[channel] = abs(float(value) - center) / mad
    if not scores:
        return {"score": 0.0, "channel": "none", "status": "no_overlap", "threshold": model.threshold}
    channel, score = max(scores.items(), key=lambda item: item[1])
    return {
        "score": round(score, 3),
        "channel": channel,
        "status": "anomaly" if score >= model.threshold else "normal",
        "threshold": model.threshold,
    }


def _read_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for channel in BASELINE_CHANNELS:
                try:
                    parsed[channel] = float(row.get(channel, 0) or 0)
                except ValueError:
                    parsed[channel] = 0.0
            rows.append(parsed)
    return rows


def _profile(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    center = median(ordered) if ordered else 0.0
    deviations = [abs(value - center) for value in ordered]
    mad = median(deviations) if deviations else 1.0
    mad = mad or 1.0
    return {
        "center": float(center),
        "mad": float(mad),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "min": float(ordered[0]) if ordered else 0.0,
        "max": float(ordered[-1]) if ordered else 0.0,
    }


def _percentile(ordered_values: list[float], quantile: float) -> float:
    if not ordered_values:
        return 0.0
    idx = min(len(ordered_values) - 1, max(0, int(round((len(ordered_values) - 1) * quantile))))
    return float(ordered_values[idx])
