from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .autotest import validate_dataset
from .baseline import load_project_baseline, score_against_baseline, train_and_save_project_baseline
from .classifier import load_project_classifier, predict_scene, train_and_save_project_classifier
from .manifest import init_project, project_path
from .models.evaluation import evaluate_project_scenes, evaluation_report_path
from .orbiters import evaluate_project_orbiters, read_recent_orbiter_summaries
from .replay import read_preview_rows
from .timeseries import (
    load_project_timeseries,
    predict_scene_timeseries,
    train_and_save_project_timeseries,
)
from .transfer import export_transfer_bundle, transfer_bundle_path
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso
from .watcher import read_recent_watcher_events, run_watcher_scan


ProgressCallback = Callable[[dict[str, object]], None]


def intelligence_state_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "intelligence_state.json"


def run_intelligence_update(
    project_name: str,
    *,
    startup: bool = False,
    run_watchers: bool = True,
    run_orbiters: bool = True,
    run_training: bool = True,
    run_transfer: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    state: dict[str, object] = {
        "format": "dsense-intelligence-state-v1",
        "project_name": project_name,
        "created_utc": utc_now_iso(),
        "startup": startup,
        "status": "ok",
        "steps": [],
        "models": {
            "baseline": {},
            "classifier": {},
            "timeseries": {},
            "watcher": {},
            "orbiters": {},
            "evaluation": {},
            "transfer": {},
        },
        "council": {
            "overall_confidence": 0.0,
            "agreement": "unknown",
            "warnings": [],
            "recommendations": [],
            "best_channels": [],
            "weak_labels": [],
        },
    }
    models = state["models"]
    assert isinstance(models, dict)

    def run_step(name: str, func: Callable[[], dict[str, object]]) -> dict[str, object]:
        step = {"name": name, "status": "running", "started_utc": utc_now_iso(), "finished_utc": "", "summary": {}}
        state["steps"].append(step)
        _notify(progress_callback, state, step)
        try:
            summary = func()
            step["status"] = str(summary.pop("status", "ok"))
            step["summary"] = summary
        except Exception as exc:
            step["status"] = "failed"
            step["summary"] = {"error": str(exc)}
        step["finished_utc"] = utc_now_iso()
        _notify(progress_callback, state, step)
        return dict(step)

    run_step("init_project", lambda: {"path": str(init_project(project_name))})
    run_step("validate", lambda: _validate_summary(project_name))

    if run_training:
        run_step("train_baseline", lambda: _baseline_summary(train_and_save_project_baseline(project_name), models))
        run_step("train_classifier", lambda: _classifier_summary(train_and_save_project_classifier(project_name), models))
        run_step("train_timeseries", lambda: _timeseries_summary(train_and_save_project_timeseries(project_name), models))
    else:
        run_step("load_models", lambda: _load_model_summaries(project_name, models))

    run_step("evaluate", lambda: _evaluation_summary(evaluate_project_scenes(project_name), models))

    if run_watchers:
        run_step("watcher", lambda: _watcher_summary(run_watcher_scan(project_name, duration=0.05, tick_hz=20), project_name, models))
    else:
        run_step("watcher", lambda: _watcher_existing_summary(project_name, models))

    if run_orbiters:
        run_step("orbiters", lambda: _orbiters_summary(evaluate_project_orbiters(project_name), project_name, models))
    else:
        run_step("orbiters", lambda: _orbiters_existing_summary(project_name, models))

    if run_transfer:
        run_step("transfer", lambda: _transfer_summary(export_transfer_bundle(project_name), models))
    else:
        run_step("transfer", lambda: _transfer_existing_summary(project_name, models))

    state["council"] = build_council_summary(project_name, models)
    state["status"] = _overall_status(list(state["steps"]), dict(state["council"]))
    run_step("write_state", lambda: _write_state_summary(project_name, state))
    state["status"] = _overall_status(list(state["steps"]), dict(state["council"]))
    write_json(intelligence_state_path(project_name), state)
    return state


def load_intelligence_state(project_name: str) -> dict[str, object] | None:
    path = intelligence_state_path(project_name)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def build_council_summary(project_name: str, models: dict[str, object] | None = None) -> dict[str, object]:
    models = models or _artifact_models(project_name)
    warnings: list[str] = []
    recommendations: list[str] = []
    weak_labels: list[str] = []
    best_channels: list[str] = []

    baseline = dict(models.get("baseline", {}))
    classifier = dict(models.get("classifier", {}))
    timeseries = dict(models.get("timeseries", {}))
    evaluation = dict(models.get("evaluation", {}))
    watcher = dict(models.get("watcher", {}))
    orbiters = dict(models.get("orbiters", {}))

    baseline_count = int(baseline.get("scene_count", 0) or 0)
    classifier_count = int(classifier.get("scene_count", 0) or 0)
    timeseries_count = int(timeseries.get("scene_count", 0) or 0)
    label_counts = {str(k): int(v) for k, v in dict(classifier.get("label_counts", timeseries.get("label_counts", {}))).items()}

    if baseline_count < 3:
        warnings.append("not enough baseline scenes")
        recommendations.append("record 3 more baseline takes or run baseline-suite to increase negative controls")
    repeated_short = sorted(label for label, count in label_counts.items() if count < 2)
    if repeated_short:
        weak_labels.extend(repeated_short)
        warnings.append("not enough repeated user labels")
        recommendations.extend(f"record 3 more takes of label {label}" for label in repeated_short[:5])
    if not watcher.get("event_count"):
        warnings.append("no watcher events available")
    if not orbiters.get("summary_count"):
        warnings.append("orbiter summaries missing")

    confusion = dict(evaluation.get("confusion_matrix", {}))
    accuracy = float(confusion.get("accuracy", 0.0) or 0.0)
    drift = dict(evaluation.get("baseline_drift", {}))
    max_drift = float(drift.get("max_drift", 0.0) or 0.0)
    if max_drift > 8:
        warnings.append("baseline drift is high")
        recommendations.append("review recent baseline scenes for machine load changes")
    elif max_drift > 3:
        warnings.append("baseline drift is elevated")

    ranking = evaluation.get("channel_usefulness_ranking", [])
    if isinstance(ranking, list):
        best_channels = [str(dict(item).get("channel", "")) for item in ranking[:5] if dict(item).get("channel")]
    if not best_channels:
        best_channels = list(timeseries.get("sequence_channels", []))[:5]
    if not best_channels:
        recommendations.append("enable linux channels for richer telemetry if available")

    agreement = _readiness_agreement(baseline_count, classifier_count, timeseries_count, label_counts)
    readiness = [
        min(1.0, baseline_count / 6.0),
        min(1.0, classifier_count / 8.0),
        min(1.0, timeseries_count / 8.0),
        accuracy if accuracy else 0.25 if classifier_count and timeseries_count else 0.0,
        max(0.0, 1.0 - min(max_drift, 10.0) / 10.0),
    ]
    confidence = sum(readiness) / len(readiness)
    if agreement == "low":
        confidence *= 0.75
    if warnings:
        confidence *= max(0.55, 1.0 - len(warnings) * 0.08)

    if accuracy and accuracy < 0.7:
        recommendations.append("review scenes with low confidence or confused labels")

    return {
        "overall_confidence": round(max(0.0, min(1.0, confidence)), 3),
        "agreement": agreement,
        "warnings": _unique(warnings),
        "recommendations": _unique(recommendations),
        "best_channels": best_channels,
        "weak_labels": _unique(weak_labels),
    }


def classify_with_council(project_name: str, scene_dir: Path) -> dict[str, object]:
    preview = scene_dir / "preview.csv"
    baseline = load_project_baseline(project_name)
    classifier = load_project_classifier(project_name)
    timeseries = load_project_timeseries(project_name)
    rows = read_preview_rows(preview) if preview.exists() else []
    latest = rows[-1] if rows else {}
    deterministic = predict_scene(classifier, preview)
    temporal = predict_scene_timeseries(timeseries, preview)
    baseline_status = score_against_baseline(_core_values(latest), baseline)
    agreement = "unknown"
    if deterministic.get("label") != "unknown" and temporal.get("label") != "unknown":
        agreement = "high" if deterministic.get("label") == temporal.get("label") else "low"
    warnings = []
    if agreement == "low":
        warnings.append("classifier and time-series disagree")
    return {
        "project_name": project_name,
        "scene_dir": str(scene_dir),
        "deterministic_classifier": deterministic,
        "time_series_classifier": temporal,
        "baseline_anomaly": baseline_status,
        "watcher_context": read_recent_watcher_events(project_name, 5),
        "orbiter_context": read_recent_orbiter_summaries(project_path(project_name), 5),
        "agreement": agreement,
        "warnings": warnings,
        "council": build_council_summary(project_name),
    }


def _validate_summary(project_name: str) -> dict[str, object]:
    result = validate_dataset(project_name)
    return {
        "status": "warning" if result.error_count or result.warning_count else "ok",
        "total_scenes": result.total_scenes,
        "valid_scenes": result.valid_scenes,
        "errors": result.error_count,
        "warnings": result.warning_count,
    }


def _baseline_summary(model, models: dict[str, object]) -> dict[str, object]:
    summary = {"scene_count": model.scene_count, "channel_count": len(model.channels), "channels": sorted(model.channels)}
    models["baseline"] = summary
    return summary


def _classifier_summary(model, models: dict[str, object]) -> dict[str, object]:
    summary = {"scene_count": model.scene_count, "baseline_scene_count": model.baseline_scene_count, "label_counts": model.label_counts}
    models["classifier"] = summary
    return summary


def _timeseries_summary(model, models: dict[str, object]) -> dict[str, object]:
    summary = {"scene_count": model.scene_count, "label_counts": model.label_counts, "sequence_channels": model.sequence_channels}
    models["timeseries"] = summary
    return summary


def _load_model_summaries(project_name: str, models: dict[str, object]) -> dict[str, object]:
    baseline = load_project_baseline(project_name)
    classifier = load_project_classifier(project_name)
    timeseries = load_project_timeseries(project_name)
    if baseline is not None:
        models["baseline"] = {"scene_count": baseline.scene_count, "channel_count": len(baseline.channels), "channels": sorted(baseline.channels)}
    if classifier is not None:
        models["classifier"] = {"scene_count": classifier.scene_count, "baseline_scene_count": classifier.baseline_scene_count, "label_counts": classifier.label_counts}
    if timeseries is not None:
        models["timeseries"] = {"scene_count": timeseries.scene_count, "label_counts": timeseries.label_counts, "sequence_channels": timeseries.sequence_channels}
    return {"loaded": True}


def _evaluation_summary(report: dict[str, object], models: dict[str, object]) -> dict[str, object]:
    summary = {
        "path": str(evaluation_report_path(str(report.get("project_name", "")))),
        "scene_count": report.get("scene_count", 0),
        "label_counts": report.get("label_counts", {}),
        "confusion_matrix": report.get("confusion_matrix", {}),
        "baseline_drift": report.get("baseline_drift", {}),
        "channel_usefulness_ranking": report.get("channel_usefulness_ranking", []),
    }
    models["evaluation"] = summary
    return summary


def _watcher_summary(result: dict[str, object], project_name: str, models: dict[str, object]) -> dict[str, object]:
    events = read_recent_watcher_events(project_name, 10)
    summary = {
        "event_count": len(events),
        "recent_events": events,
        "last_scene_id": dict(result.get("scene", {})).get("scene_id"),
        "path": result.get("watcher_events_path", ""),
    }
    models["watcher"] = summary
    return summary


def _watcher_existing_summary(project_name: str, models: dict[str, object]) -> dict[str, object]:
    events = read_recent_watcher_events(project_name, 10)
    summary = {"status": "skipped", "event_count": len(events), "recent_events": events}
    models["watcher"] = summary
    return summary


def _orbiters_summary(report: dict[str, object], project_name: str, models: dict[str, object]) -> dict[str, object]:
    summaries = read_recent_orbiter_summaries(project_path(project_name), 10)
    summary = {"evaluated": report.get("evaluated", 0), "accuracy": report.get("accuracy", 0.0), "summary_count": len(summaries), "recent_summaries": summaries}
    models["orbiters"] = summary
    return summary


def _orbiters_existing_summary(project_name: str, models: dict[str, object]) -> dict[str, object]:
    summaries = read_recent_orbiter_summaries(project_path(project_name), 10)
    summary = {"status": "skipped", "summary_count": len(summaries), "recent_summaries": summaries}
    models["orbiters"] = summary
    return summary


def _transfer_summary(bundle: dict[str, object], models: dict[str, object]) -> dict[str, object]:
    summary = {"path": str(transfer_bundle_path(str(bundle.get("project_name", "")))), "total_scenes": bundle.get("total_scenes", 0), "scene_counts": bundle.get("scene_counts", {})}
    models["transfer"] = summary
    return summary


def _transfer_existing_summary(project_name: str, models: dict[str, object]) -> dict[str, object]:
    path = transfer_bundle_path(project_name)
    summary: dict[str, object] = {"status": "skipped", "path": str(path), "exists": path.exists()}
    models["transfer"] = summary
    return summary


def _write_state_summary(project_name: str, state: dict[str, object]) -> dict[str, object]:
    path = intelligence_state_path(project_name)
    ensure_dir(path.parent)
    write_json(path, state)
    return {"path": str(path)}


def _artifact_models(project_name: str) -> dict[str, object]:
    models: dict[str, object] = {}
    _load_model_summaries(project_name, models)
    evaluation_path = evaluation_report_path(project_name)
    if evaluation_path.exists():
        try:
            models["evaluation"] = read_json(evaluation_path)
        except (OSError, ValueError):
            pass
    _watcher_existing_summary(project_name, models)
    _orbiters_existing_summary(project_name, models)
    _transfer_existing_summary(project_name, models)
    return models


def _overall_status(steps: list[object], council: dict[str, object]) -> str:
    statuses = {str(dict(step).get("status", "")) for step in steps if isinstance(step, dict)}
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses or dict(council).get("warnings"):
        return "warning"
    return "ok"


def _readiness_agreement(baseline_count: int, classifier_count: int, timeseries_count: int, label_counts: dict[str, int]) -> str:
    if classifier_count == 0 or timeseries_count == 0:
        return "unknown"
    repeated = sum(1 for count in label_counts.values() if count >= 2)
    if baseline_count >= 3 and repeated >= 2 and classifier_count == timeseries_count:
        return "high"
    if baseline_count >= 1 and repeated >= 1:
        return "medium"
    return "low"


def _notify(callback: ProgressCallback | None, state: dict[str, object], step: dict[str, object]) -> None:
    if callback is not None:
        callback({"state": state, "step": dict(step)})


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _core_values(row: dict[str, object]) -> dict[str, float]:
    def number(value: object) -> float:
        try:
            return float(value) if value not in (None, "") else 0.0
        except (TypeError, ValueError):
            return 0.0

    return {
        "dt_ns": number(row.get("dt_ns")),
        "sleep_drift_ns": abs(number(row.get("sleep_drift_ns"))),
        "process_ns_estimate": number(row.get("process_ns_estimate")),
    }
