from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from .manifest import project_path
from .models.evaluation import predict_from_profiles
from .models.features import feature_distance, feature_manifest, mean_profile, read_numeric_preview_rows, slope, variance
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso


@dataclass(frozen=True)
class TimeSeriesModel:
    project_name: str
    trained_utc: str
    scene_count: int
    label_counts: dict[str, int]
    label_profiles: dict[str, dict[str, float]]
    sequence_channels: list[str]
    feature_manifest: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "dsense-timeseries-model-v1",
            "project_name": self.project_name,
            "trained_utc": self.trained_utc,
            "scene_count": self.scene_count,
            "label_counts": self.label_counts,
            "label_profiles": self.label_profiles,
            "sequence_channels": self.sequence_channels,
            "feature_manifest": self.feature_manifest,
        }


def timeseries_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "timeseries_model.json"


def train_project_timeseries(project_name: str) -> TimeSeriesModel:
    root = project_path(project_name)
    label_counts: dict[str, int] = {}
    label_features: dict[str, list[dict[str, float]]] = {}
    all_features: list[dict[str, float]] = []
    all_rows: list[dict[str, float]] = []
    sequence_channels: set[str] = set()
    scene_count = 0

    for scene_path in sorted((root / "scenes").glob("scene_*/scene.json")):
        try:
            scene = read_json(scene_path)
        except (OSError, ValueError):
            continue
        if scene.get("accepted") is False:
            continue
        preview_path = scene_path.parent / "preview.csv"
        if not preview_path.exists():
            continue
        rows = read_numeric_preview_rows(preview_path)
        if not rows:
            continue
        label = str(scene.get("label", "unknown"))
        features = extract_timeseries_features(rows)
        if not features:
            continue
        scene_count += 1
        all_features.append(features)
        all_rows.extend(rows)
        sequence_channels.update({channel for row in rows for channel in row})
        label_counts[label] = label_counts.get(label, 0) + 1
        label_features.setdefault(label, []).append(features)

    profiles = {label: mean_profile(features) for label, features in label_features.items()}
    manifest = feature_manifest(all_features, all_rows)
    manifest["timeseries_stats"] = [
        "first",
        "last",
        "slope",
        "peak_count",
        "roughness",
        "rolling_variance",
        "max_abs_delta",
        "window_median",
    ]
    return TimeSeriesModel(
        project_name=project_name,
        trained_utc=utc_now_iso(),
        scene_count=scene_count,
        label_counts=label_counts,
        label_profiles=profiles,
        sequence_channels=sorted(sequence_channels),
        feature_manifest=manifest,
    )


def train_and_save_project_timeseries(project_name: str) -> TimeSeriesModel:
    model = train_project_timeseries(project_name)
    out = timeseries_path(project_name)
    ensure_dir(out.parent)
    write_json(out, model.to_dict())
    return model


def load_project_timeseries(project_name: str) -> TimeSeriesModel | None:
    path = timeseries_path(project_name)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    return TimeSeriesModel(
        project_name=str(data.get("project_name", project_name)),
        trained_utc=str(data.get("trained_utc", "")),
        scene_count=int(data.get("scene_count", 0)),
        label_counts={str(k): int(v) for k, v in dict(data.get("label_counts", {})).items()},
        label_profiles={
            str(label): {str(k): float(v) for k, v in dict(profile).items()}
            for label, profile in dict(data.get("label_profiles", {})).items()
        },
        sequence_channels=[str(channel) for channel in list(data.get("sequence_channels", []))],
        feature_manifest=dict(data.get("feature_manifest", {})),
    )


def predict_scene_timeseries(model: TimeSeriesModel | None, preview_path: Path) -> dict[str, object]:
    rows = read_numeric_preview_rows(preview_path) if preview_path.exists() else []
    if model is None or not model.label_profiles or not rows:
        return {"label": "unknown", "confidence": 0.0, "distance": 0.0, "contributions": {}, "sequence_channels": []}
    features = extract_timeseries_features(rows)
    prediction = predict_from_profiles(model.label_profiles, features)
    prediction["sequence_channels"] = model.sequence_channels
    return prediction


def extract_timeseries_features(rows: list[dict[str, float]], windows: int = 4) -> dict[str, float]:
    features: dict[str, float] = {}
    channels = sorted({channel for row in rows for channel in row})
    for channel in channels:
        values = [float(row.get(channel, 0.0)) for row in rows]
        if not values:
            continue
        deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
        abs_deltas = [abs(delta) for delta in deltas]
        peaks = _peak_count(values)
        features[f"{channel}_first"] = values[0]
        features[f"{channel}_last"] = values[-1]
        features[f"{channel}_slope"] = slope(values)
        features[f"{channel}_peak_count"] = float(peaks)
        features[f"{channel}_roughness"] = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
        features[f"{channel}_rolling_variance"] = _rolling_variance(values)
        features[f"{channel}_max_abs_delta"] = max(abs_deltas) if abs_deltas else 0.0
        for window_index, window_values in enumerate(_fixed_windows(values, windows)):
            features[f"{channel}_window_{window_index}_median"] = float(median(window_values)) if window_values else 0.0
    return features


def compare_timeseries_profiles(left: dict[str, float], right: dict[str, float]) -> tuple[float, dict[str, float]]:
    return feature_distance(left, right)


def _fixed_windows(values: list[float], windows: int) -> list[list[float]]:
    count = max(1, windows)
    if len(values) <= count:
        return [[value] for value in values]
    out: list[list[float]] = []
    for index in range(count):
        start = int(round(index * len(values) / count))
        end = int(round((index + 1) * len(values) / count))
        out.append(values[start:max(start + 1, end)])
    return out


def _rolling_variance(values: list[float], window: int = 5) -> float:
    if not values:
        return 0.0
    width = max(2, min(window, len(values)))
    variances = [variance(values[index:index + width]) for index in range(0, len(values) - width + 1)]
    return float(sum(variances) / len(variances)) if variances else 0.0


def _peak_count(values: list[float]) -> int:
    if len(values) < 3:
        return 0
    count = 0
    for previous, current, following in zip(values, values[1:], values[2:]):
        if current > previous and current > following:
            count += 1
    return count
