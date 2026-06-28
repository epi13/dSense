from __future__ import annotations

from pathlib import Path

from .baseline import load_project_baseline, train_and_save_project_baseline
from .classifier import load_project_classifier, train_and_save_project_classifier
from .manifest import project_path, scan_channels
from .privacy import redact_transfer_bundle
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso


def transfer_bundle_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "transfer_bundle.json"


def export_transfer_bundle(project_name: str, redact: bool = False) -> dict[str, object]:
    baseline = load_project_baseline(project_name) or train_and_save_project_baseline(project_name)
    classifier = load_project_classifier(project_name) or train_and_save_project_classifier(project_name)
    scenes = _load_project_scenes(project_name)
    bundle = {
        "format": "dsense-transfer-v1",
        "created_utc": utc_now_iso(),
        "project_name": project_name,
        "scene_counts": _scene_counts(scenes),
        "total_scenes": len(scenes),
        "baseline": baseline.to_dict(),
        "classifier": classifier.to_dict(),
        "channels": scan_channels(),
        "quality": _quality_summary(scenes),
    }
    if redact:
        bundle = redact_transfer_bundle(bundle)
    out = transfer_bundle_path(project_name)
    ensure_dir(out.parent)
    write_json(out, bundle)
    return bundle


def compare_transfer_bundle(project_name: str, bundle_path: Path) -> dict[str, object]:
    local = export_transfer_bundle(project_name, redact=False)
    remote = read_json(bundle_path)
    local_channels = {channel["id"] for channel in local.get("channels", []) if channel.get("available")}
    remote_channels = {channel["id"] for channel in remote.get("channels", []) if channel.get("available")}
    missing = sorted(remote_channels - local_channels)
    extra = sorted(local_channels - remote_channels)
    local_baseline = dict(local.get("baseline", {})).get("channels", {})
    remote_baseline = dict(remote.get("baseline", {})).get("channels", {})
    drift = _baseline_drift(local_baseline, remote_baseline)
    risk = "low"
    if missing or drift.get("max_drift", 0) > 3:
        risk = "medium"
    if len(missing) > 2 or drift.get("max_drift", 0) > 8:
        risk = "high"
    return {
        "project_name": project_name,
        "bundle": str(bundle_path),
        "compatibility": "compatible" if not missing else "degraded",
        "missing_channels": missing,
        "extra_channels": extra,
        "baseline_drift": drift,
        "transfer_risk": risk,
    }


def _quality_summary(scenes: list[dict[str, object]]) -> dict[str, float]:
    confidences = [
        float(scene.get("quality", {}).get("confidence", 0.0))
        for scene in scenes
        if isinstance(scene.get("quality"), dict)
    ]
    if not confidences:
        return {"min_confidence": 0.0, "avg_confidence": 0.0}
    return {"min_confidence": min(confidences), "avg_confidence": sum(confidences) / len(confidences)}


def _load_project_scenes(project_name: str) -> list[dict[str, object]]:
    scenes = []
    for path in sorted((project_path(project_name) / "scenes").glob("scene_*/scene.json")):
        try:
            scenes.append(read_json(path))
        except (OSError, ValueError):
            continue
    return scenes


def _scene_counts(scenes: list[dict[str, object]]) -> dict[str, int]:
    counts = {"baseline": 0, "user": 0, "other": 0}
    for scene in scenes:
        label = str(scene.get("label", ""))
        if label.startswith("baseline_"):
            counts["baseline"] += 1
        elif label.startswith("user_") or label.startswith("person_"):
            counts["user"] += 1
        else:
            counts["other"] += 1
    return counts


def _baseline_drift(local: object, remote: object) -> dict[str, object]:
    local_profiles = dict(local)
    remote_profiles = dict(remote)
    scores = {}
    for channel in sorted(set(local_profiles) & set(remote_profiles)):
        l_profile = dict(local_profiles[channel])
        r_profile = dict(remote_profiles[channel])
        denom = max(float(l_profile.get("mad", 1.0)), float(r_profile.get("mad", 1.0)), 1.0)
        scores[channel] = abs(float(l_profile.get("center", 0.0)) - float(r_profile.get("center", 0.0))) / denom
    return {"channels": scores, "max_drift": max(scores.values()) if scores else 0.0}
