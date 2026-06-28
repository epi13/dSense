from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .manifest import project_path
from .models.features import feature_manifest, full_profile, percentile, read_numeric_preview_rows, summarize_rows
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
    feature_manifest: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_name": self.project_name,
            "trained_utc": self.trained_utc,
            "scene_count": self.scene_count,
            "threshold": self.threshold,
            "channels": self.channels,
            "feature_manifest": self.feature_manifest,
        }


def baseline_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "baseline_model.json"


def train_project_baseline(project_name: str, threshold: float = 6.0) -> BaselineModel:
    root = project_path(project_name)
    values: dict[str, list[float]] = {}
    all_features: list[dict[str, float]] = []
    all_rows: list[dict[str, float]] = []
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
        all_features.append(summarize_rows(rows))
        all_rows.extend(rows)
        scene_count += 1
        for row in rows:
            for channel, value in row.items():
                values.setdefault(channel, []).append(abs(float(value)))

    channels = {
        channel: _profile(channel_values)
        for channel, channel_values in values.items()
        if channel_values
    }
    return BaselineModel(project_name, utc_now_iso(), scene_count, threshold, channels, feature_manifest(all_features, all_rows))


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
        feature_manifest=dict(data.get("feature_manifest", {})),
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
    return read_numeric_preview_rows(path)


def _profile(values: list[float]) -> dict[str, float]:
    return full_profile(values)


def _percentile(ordered_values: list[float], quantile: float) -> float:
    return percentile(ordered_values, quantile)
