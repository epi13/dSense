from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import project_path
from .models.features import feature_manifest, full_profile, percentile, read_numeric_preview_rows, summarize_rows
from .models.scene_store import SceneFeatureStore, build_or_load_feature_store, feature_manifest_from_store
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso

BASELINE_CHANNELS = ("dt_ns", "sleep_drift_ns", "process_ns_estimate")
BASELINE_MODEL_VERSION = "baseline-profile-v2"


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
    store = build_or_load_feature_store(project_name, workers=1)
    return train_project_baseline_from_store(store, threshold=threshold)


def train_project_baseline_from_store(store: SceneFeatureStore, threshold: float = 6.0) -> BaselineModel:
    values: dict[str, list[float]] = {}
    all_features: list[dict[str, float]] = []
    all_rows: list[dict[str, float]] = []
    scene_count = 0
    for scene in store.accepted_scenes:
        if not scene.label.startswith("baseline_"):
            continue
        rows = scene.preview_rows
        if not rows:
            continue
        all_features.append(scene.summary_features or summarize_rows(rows))
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
    manifest = feature_manifest_from_store(store)
    if not manifest.get("features"):
        manifest = feature_manifest(all_features, all_rows)
    manifest["model_version"] = BASELINE_MODEL_VERSION
    manifest["dataset_fingerprint"] = store.fingerprint
    return BaselineModel(store.project_name, utc_now_iso(), scene_count, threshold, channels, manifest)


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


def project_has_usable_baseline(project_name: str) -> bool:
    model = load_project_baseline(project_name)
    return model is not None and model.scene_count > 0 and bool(model.channels)


def default_auto_baseline_policy() -> str:
    return "startup" if platform.system().lower() == "linux" else "missing-only"


def ensure_startup_baseline(
    project_name: str,
    duration: float = 5.0,
    tick_hz: int = 100,
    policy: str = "auto",
    force: bool = False,
) -> dict[str, object]:
    from .manifest import allocate_scene_id, init_project
    from .recorder import record_scene

    init_project(project_name)
    resolved_policy = default_auto_baseline_policy() if policy == "auto" else policy
    if resolved_policy not in {"startup", "missing-only", "off"}:
        raise ValueError(f"Unknown auto-baseline policy: {policy}")
    if resolved_policy == "off":
        return {
            "status": "skipped",
            "recorded": False,
            "scene_id": None,
            "policy": resolved_policy,
            "message": "Startup baseline: skipped by policy off",
        }
    if not force and resolved_policy == "missing-only" and project_has_usable_baseline(project_name):
        return {
            "status": "reused",
            "recorded": False,
            "scene_id": None,
            "policy": resolved_policy,
            "message": "Startup baseline: reused existing model",
        }

    try:
        scene_id = allocate_scene_id(project_name)
        scene_dir = project_path(project_name) / "scenes" / scene_id
        scene = record_scene(
            scene_dir,
            scene_id,
            "baseline_startup_auto",
            max(0.01, duration),
            max(1, tick_hz),
            0.0,
            max(0.01, duration),
            0.0,
            "Automatically recorded startup baseline when dSense TUI opened.",
            mode="baseline_auto",
            channel_groups=("portable", "linux"),
        )
        model = train_and_save_project_baseline(project_name)
    except Exception as exc:
        return {
            "status": "failed",
            "recorded": False,
            "scene_id": None,
            "policy": resolved_policy,
            "message": f"Startup baseline: failed: {exc}",
        }

    return {
        "status": "recorded",
        "recorded": True,
        "scene_id": scene_id,
        "scene": scene,
        "policy": resolved_policy,
        "baseline_scene_count": model.scene_count,
        "channel_count": len(model.channels),
        "message": f"Startup baseline: recorded {scene_id}",
    }


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
