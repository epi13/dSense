from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .manifest import project_path
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso

CHANNELS = ("dt_ns", "sleep_drift_ns", "process_ns_estimate")


@dataclass(frozen=True)
class SceneClassifierModel:
    project_name: str
    trained_utc: str
    scene_count: int
    baseline_scene_count: int
    label_counts: dict[str, int]
    detector_baseline: dict[str, dict[str, float]]
    label_profiles: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "project_name": self.project_name,
            "trained_utc": self.trained_utc,
            "scene_count": self.scene_count,
            "baseline_scene_count": self.baseline_scene_count,
            "label_counts": self.label_counts,
            "detector_baseline": self.detector_baseline,
            "label_profiles": self.label_profiles,
        }


def train_project_classifier(project_name: str) -> SceneClassifierModel:
    root = project_path(project_name)
    scene_rows: list[dict[str, object]] = []
    baseline_rows: dict[str, list[float]] = {channel: [] for channel in CHANNELS}
    label_counts: dict[str, int] = {}
    label_features: dict[str, list[dict[str, float]]] = {}
    baseline_scene_count = 0

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

        label = str(scene.get("label", "unknown"))
        rows = _read_preview_rows(preview_path)
        if not rows:
            continue

        features = _summarize_rows(rows)
        scene_rows.append({"scene_id": scene.get("scene_id"), "label": label, "features": features})
        label_counts[label] = label_counts.get(label, 0) + 1
        label_features.setdefault(label, []).append(features)

        if label.startswith("baseline_"):
            baseline_scene_count += 1
            for row in rows:
                for channel in CHANNELS:
                    baseline_rows[channel].append(abs(float(row[channel])))

    detector_baseline = {
        channel: _robust_profile(values)
        for channel, values in baseline_rows.items()
        if values
    }
    label_profiles = {
        label: _mean_profile(features)
        for label, features in label_features.items()
    }

    return SceneClassifierModel(
        project_name=project_name,
        trained_utc=utc_now_iso(),
        scene_count=len(scene_rows),
        baseline_scene_count=baseline_scene_count,
        label_counts=label_counts,
        detector_baseline=detector_baseline,
        label_profiles=label_profiles,
    )


def train_and_save_project_classifier(project_name: str) -> SceneClassifierModel:
    model = train_project_classifier(project_name)
    out = classifier_path(project_name)
    ensure_dir(out.parent)
    write_json(out, model.to_dict())
    return model


def load_project_classifier(project_name: str) -> SceneClassifierModel | None:
    path = classifier_path(project_name)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    return SceneClassifierModel(
        project_name=str(data.get("project_name", project_name)),
        trained_utc=str(data.get("trained_utc", "")),
        scene_count=int(data.get("scene_count", 0)),
        baseline_scene_count=int(data.get("baseline_scene_count", 0)),
        label_counts={str(k): int(v) for k, v in dict(data.get("label_counts", {})).items()},
        detector_baseline={
            str(channel): {str(k): float(v) for k, v in dict(profile).items()}
            for channel, profile in dict(data.get("detector_baseline", {})).items()
        },
        label_profiles={
            str(label): {str(k): float(v) for k, v in dict(profile).items()}
            for label, profile in dict(data.get("label_profiles", {})).items()
        },
    )


def classifier_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "classifier.json"


def predict_features(model: SceneClassifierModel | None, features: dict[str, float]) -> dict[str, object]:
    if model is None or not model.label_profiles:
        return {"label": "unknown", "confidence": 0.0, "distance": 0.0, "contributions": {}}
    distances = []
    for label, profile in model.label_profiles.items():
        shared = sorted(set(features) & set(profile))
        if not shared:
            continue
        contributions = {
            key: abs(float(features.get(key, 0.0)) - float(profile.get(key, 0.0))) / max(abs(float(profile.get(key, 0.0))), 1.0)
            for key in shared
        }
        distance = sum(contributions.values()) / len(contributions)
        distances.append((distance, label, contributions))
    if not distances:
        return {"label": "unknown", "confidence": 0.0, "distance": 0.0, "contributions": {}}
    distance, label, contributions = min(distances, key=lambda item: (item[0], item[1]))
    confidence = round(1.0 / (1.0 + distance), 3)
    top_contrib = dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:5])
    return {"label": label, "confidence": confidence, "distance": round(distance, 6), "contributions": top_contrib}


def predict_scene(model: SceneClassifierModel | None, preview_path: Path) -> dict[str, object]:
    rows = _read_preview_rows(preview_path)
    if not rows:
        return {"label": "unknown", "confidence": 0.0, "distance": 0.0, "contributions": {}}
    return predict_features(model, _summarize_rows(rows))


def _read_preview_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append({channel: float(row.get(channel, 0) or 0) for channel in CHANNELS})
            except ValueError:
                continue
    return rows


def _summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    features: dict[str, float] = {}
    for channel in CHANNELS:
        values = [abs(float(row[channel])) for row in rows]
        profile = _robust_profile(values)
        features[f"{channel}_median"] = profile["center"]
        features[f"{channel}_mad"] = profile["mad"]
        features[f"{channel}_p95"] = _percentile(values, 0.95)
    return features


def _robust_profile(values: list[float]) -> dict[str, float]:
    if not values:
        return {"center": 0.0, "mad": 1.0}
    center = median(values)
    deviations = [abs(value - center) for value in values]
    mad = median(deviations) or 1.0
    return {"center": float(center), "mad": float(mad)}


def _mean_profile(features: list[dict[str, float]]) -> dict[str, float]:
    if not features:
        return {}
    keys = sorted({key for feature in features for key in feature})
    return {
        key: sum(feature.get(key, 0.0) for feature in features) / len(features)
        for key in keys
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return float(ordered[idx])
