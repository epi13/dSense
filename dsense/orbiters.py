from __future__ import annotations

import json
from pathlib import Path

from .baseline import load_project_baseline, score_against_baseline, train_and_save_project_baseline
from .classifier import load_project_classifier, predict_scene, train_and_save_project_classifier
from .gemma_edge import enrich_orbiter_summary
from .manifest import project_path
from .privacy import build_privacy_report
from .replay import inspect_scene, read_preview_rows, resolve_scene_dir
from .utils.files import ensure_dir
from .utils.timebase import utc_now_iso


def orbiter_dir(project_root: Path) -> Path:
    return project_root / "exports" / "orbiters"


def make_orbiter_summary(
    scene_id: str,
    baseline_status: dict[str, object],
    classifier_prediction: dict[str, object],
    channel_availability: int,
    quality_flags: int,
    anomaly_score: float,
) -> dict[str, object]:
    label = classifier_prediction.get("label", "unknown")
    confidence = classifier_prediction.get("confidence", 0.0)
    status = baseline_status.get("status", "unknown")
    channel = baseline_status.get("channel", "none")
    summary = f"{status} substrate window; strongest channel {channel}; classifier suggests {label} ({confidence}). Evidence only; no human certainty."
    orbiter_types = _structured_orbiter_types(
        baseline_status,
        classifier_prediction,
        channel_availability,
        quality_flags,
        anomaly_score,
        privacy_context={},
        transfer_context={},
    )
    return {
        "schema_version": "dsense-orbiter-v1",
        "created_utc": utc_now_iso(),
        "scene_id": scene_id,
        "baseline_status": baseline_status,
        "classifier_prediction": classifier_prediction,
        "anomaly_score": anomaly_score,
        "channel_availability_mask": channel_availability,
        "quality_flags": quality_flags,
        "orbiter_types": orbiter_types,
        "local_model_adapters": local_model_adapters(),
        "confidence_disclaimer": confidence_disclaimer(),
        "summary": summary,
    }


def run_scene_orbiters(project_name: str, scene_id: str) -> dict[str, object]:
    scene_dir = resolve_scene_dir(project_name, scene_id)
    baseline = load_project_baseline(project_name) or train_and_save_project_baseline(project_name)
    classifier = load_project_classifier(project_name) or train_and_save_project_classifier(project_name)
    preview = scene_dir / "preview.csv"
    rows = read_preview_rows(preview)
    latest = rows[-1] if rows else {}
    baseline_status = score_against_baseline(_core_values(latest), baseline)
    classifier_prediction = predict_scene(classifier, preview)
    scene_summary = inspect_scene(scene_dir)
    actual_label = str(scene_summary.get("label", "unknown"))
    prediction_label = str(classifier_prediction.get("label", "unknown"))
    privacy_report = build_privacy_report(project_name)
    anomaly_score = float(baseline_status.get("score", 0.0) or 0.0)
    summary = make_orbiter_summary(
        scene_id,
        baseline_status,
        classifier_prediction,
        _availability_mask(scene_summary),
        _int_value(latest.get("quality_flags"), 0),
        anomaly_score,
    )
    summary["actual_label"] = actual_label
    summary["summary_comparison"] = {
        "actual_label": actual_label,
        "predicted_label": prediction_label,
        "matches_actual_label": actual_label == prediction_label,
        "confidence": classifier_prediction.get("confidence", 0.0),
    }
    summary["scene"] = scene_summary
    summary["orbiter_types"] = _structured_orbiter_types(
        baseline_status,
        classifier_prediction,
        _availability_mask(scene_summary),
        _int_value(latest.get("quality_flags"), 0),
        anomaly_score,
        privacy_context=privacy_report,
        transfer_context={"project_name": project_name, "scene_count": privacy_report.get("scene_count", 0)},
    )
    return enrich_orbiter_summary(summary)


def evaluate_project_orbiters(project_name: str) -> dict[str, object]:
    root = project_path(project_name)
    summaries = []
    correct = 0
    evaluated = 0
    for scene_path in sorted((root / "scenes").glob("scene_*/scene.json")):
        scene_id = scene_path.parent.name
        summary = run_scene_orbiters(project_name, scene_id)
        comparison = dict(summary.get("summary_comparison", {}))
        summaries.append({
            "scene_id": scene_id,
            "actual_label": comparison.get("actual_label", "unknown"),
            "predicted_label": comparison.get("predicted_label", "unknown"),
            "matches_actual_label": comparison.get("matches_actual_label", False),
            "confidence": comparison.get("confidence", 0.0),
            "disclaimer": summary.get("confidence_disclaimer", ""),
        })
        evaluated += 1
        if comparison.get("matches_actual_label"):
            correct += 1
    return {
        "format": "dsense-orbiter-evaluation-v1",
        "created_utc": utc_now_iso(),
        "project_name": project_name,
        "evaluated": evaluated,
        "matches": correct,
        "accuracy": round(correct / evaluated, 6) if evaluated else 0.0,
        "summaries": summaries,
    }


def append_orbiter_summary(project_root: Path, summary: dict[str, object]) -> Path:
    summary = enrich_orbiter_summary(summary)
    out_dir = orbiter_dir(project_root)
    ensure_dir(out_dir)
    path = out_dir / "summaries.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")
    return path


def read_recent_orbiter_summaries(project_root: Path, limit: int = 5) -> list[dict[str, object]]:
    path = orbiter_dir(project_root) / "summaries.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows[-limit:]


def local_model_adapters() -> dict[str, object]:
    from .gemma_edge import gemma_edge_status

    return {
        "gemma": gemma_edge_status(),
        "onnx": {"enabled": False, "mode": "local_only_optional_adapter_not_configured"},
        "tiny_classifier": {"enabled": True, "mode": "deterministic_nearest_profile"},
        "remote_calls": False,
    }


def confidence_disclaimer() -> str:
    return "Orbiter summaries describe substrate evidence only; they do not prove human presence, identity, intent, camera, microphone, or RF observations."


def _structured_orbiter_types(
    baseline_status: dict[str, object],
    classifier_prediction: dict[str, object],
    channel_availability: int,
    quality_flags: int,
    anomaly_score: float,
    privacy_context: dict[str, object],
    transfer_context: dict[str, object],
) -> dict[str, object]:
    return {
        "timing": {
            "status": baseline_status.get("status", "unknown"),
            "strongest_channel": baseline_status.get("channel", "none"),
            "score": baseline_status.get("score", 0.0),
            "evidence": ["baseline robust score", "channel availability mask", "quality flags"],
            "confidence": _bounded_confidence(anomaly_score),
            "disclaimer": confidence_disclaimer(),
        },
        "activity": {
            "predicted_label": classifier_prediction.get("label", "unknown"),
            "confidence": classifier_prediction.get("confidence", 0.0),
            "distance": classifier_prediction.get("distance", 0.0),
            "evidence": classifier_prediction.get("contributions", {}),
            "disclaimer": confidence_disclaimer(),
        },
        "drift": {
            "status": baseline_status.get("status", "unknown"),
            "threshold": baseline_status.get("threshold", 0.0),
            "anomaly_score": anomaly_score,
            "evidence": {"channel_availability_mask": channel_availability, "quality_flags": quality_flags},
            "disclaimer": "Drift is machine-relative and may reflect scheduler/load changes rather than external activity.",
        },
        "privacy": {
            "warnings": privacy_context.get("warnings", []),
            "sensitive_labels": privacy_context.get("sensitive_labels", []),
            "sensitive_channels": privacy_context.get("sensitive_channels", []),
            "disclaimer": "Labels, timing, and channel availability may identify routines or machines.",
        },
        "transfer": {
            "project_name": transfer_context.get("project_name", "unknown"),
            "scene_count": transfer_context.get("scene_count", 0),
            "evidence": "local model statistics only unless explicitly exported",
            "disclaimer": "Transfer summaries are compatibility hints, not cross-machine guarantees.",
        },
    }


def _core_values(row: dict[str, object]) -> dict[str, float]:
    return {
        "dt_ns": _float_value(row.get("dt_ns"), 0.0),
        "sleep_drift_ns": abs(_float_value(row.get("sleep_drift_ns"), 0.0)),
        "process_ns_estimate": _float_value(row.get("process_ns_estimate"), 0.0),
    }


def _availability_mask(scene_summary: dict[str, object]) -> int:
    mask = 0
    for channel in scene_summary.get("channels", []):
        if isinstance(channel, dict) and channel.get("available"):
            mask |= 1 << int(channel.get("bit", 0) or 0)
    return mask


def _bounded_confidence(score: float) -> float:
    return round(max(0.0, min(1.0, score / 12.0)), 3)


def _float_value(value: object, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _int_value(value: object, default: int) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default
