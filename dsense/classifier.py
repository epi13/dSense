from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .manifest import project_path
from .models.evaluation import predict_from_profiles
from .models.features import feature_manifest, mean_profile, percentile, read_numeric_preview_rows, robust_profile, summarize_rows
from .models.scene_store import SceneFeatureStore, build_or_load_feature_store, feature_manifest_from_store
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso

CHANNELS = ("dt_ns", "sleep_drift_ns", "process_ns_estimate")
CLASSIFIER_MODEL_VERSION = "classifier-profile-v2"


@dataclass(frozen=True)
class SceneClassifierModel:
    project_name: str
    trained_utc: str
    scene_count: int
    baseline_scene_count: int
    label_counts: dict[str, int]
    detector_baseline: dict[str, dict[str, float]]
    label_profiles: dict[str, dict[str, float]]
    feature_manifest: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_name": self.project_name,
            "trained_utc": self.trained_utc,
            "scene_count": self.scene_count,
            "baseline_scene_count": self.baseline_scene_count,
            "label_counts": self.label_counts,
            "detector_baseline": self.detector_baseline,
            "label_profiles": self.label_profiles,
            "feature_manifest": self.feature_manifest,
        }


def train_project_classifier(project_name: str) -> SceneClassifierModel:
    store = build_or_load_feature_store(project_name, workers=1)
    return train_project_classifier_from_store(store)


def train_project_classifier_from_store(store: SceneFeatureStore) -> SceneClassifierModel:
    scene_rows: list[dict[str, object]] = []
    baseline_rows: dict[str, list[float]] = {}
    label_counts: dict[str, int] = {}
    label_features: dict[str, list[dict[str, float]]] = {}
    all_features: list[dict[str, float]] = []
    all_rows: list[dict[str, float]] = []
    baseline_scene_count = 0

    for scene in store.accepted_scenes:
        label = scene.label
        rows = scene.preview_rows
        if not rows:
            continue

        features = scene.summary_features or _summarize_rows(rows)
        all_features.append(features)
        all_rows.extend(rows)
        scene_rows.append({"scene_id": scene.scene_id, "label": label, "features": features})
        label_counts[label] = label_counts.get(label, 0) + 1
        label_features.setdefault(label, []).append(features)

        if label.startswith("baseline_"):
            baseline_scene_count += 1
            for row in rows:
                for channel, value in row.items():
                    baseline_rows.setdefault(channel, []).append(abs(float(value)))

    detector_baseline = {
        channel: robust_profile(values)
        for channel, values in baseline_rows.items()
        if values
    }
    label_profiles = {
        label: _mean_profile(features)
        for label, features in label_features.items()
    }

    return SceneClassifierModel(
        project_name=store.project_name,
        trained_utc=utc_now_iso(),
        scene_count=len(scene_rows),
        baseline_scene_count=baseline_scene_count,
        label_counts=label_counts,
        detector_baseline=detector_baseline,
        label_profiles=label_profiles,
        feature_manifest=_classifier_manifest(store, all_features, all_rows),
    )


def _classifier_manifest(store: SceneFeatureStore, all_features: list[dict[str, float]], all_rows: list[dict[str, float]]) -> dict[str, object]:
    manifest = feature_manifest_from_store(store)
    if not manifest.get("features"):
        manifest = feature_manifest(all_features, all_rows)
    manifest["model_version"] = CLASSIFIER_MODEL_VERSION
    manifest["dataset_fingerprint"] = store.fingerprint
    return manifest


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
        feature_manifest=dict(data.get("feature_manifest", {})),
    )


def classifier_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "classifier.json"


def predict_features(model: SceneClassifierModel | None, features: dict[str, float]) -> dict[str, object]:
    if model is None or not model.label_profiles:
        return {"label": "unknown", "confidence": 0.0, "distance": 0.0, "contributions": {}}
    return predict_from_profiles(model.label_profiles, features)


def predict_scene(model: SceneClassifierModel | None, preview_path: Path) -> dict[str, object]:
    rows = _read_preview_rows(preview_path)
    if not rows:
        return {"label": "unknown", "confidence": 0.0, "distance": 0.0, "contributions": {}}
    return predict_features(model, _summarize_rows(rows))


def _read_preview_rows(path: Path) -> list[dict[str, float]]:
    return read_numeric_preview_rows(path)


def _summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return summarize_rows(rows)


def _robust_profile(values: list[float]) -> dict[str, float]:
    return robust_profile(values)


def _mean_profile(features: list[dict[str, float]]) -> dict[str, float]:
    return mean_profile(features)


def _percentile(values: list[float], quantile: float) -> float:
    return percentile(values, quantile)
