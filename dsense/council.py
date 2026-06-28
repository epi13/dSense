from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from .autotest import validate_dataset
from .baseline import load_project_baseline, score_against_baseline, train_and_save_project_baseline
from .classifier import load_project_classifier, predict_scene, train_and_save_project_classifier
from .manifest import init_project, project_path
from .models.evaluation import evaluate_project_scenes, evaluation_report_path
from .orbiters import evaluate_project_orbiters, read_recent_orbiter_summaries, update_project_orbiters_incremental
from .replay import read_preview_rows
from .scenarios import all_scenarios
from .timeseries import (
    load_project_timeseries,
    predict_scene_timeseries,
    train_and_save_project_timeseries,
)
from .transfer import export_transfer_bundle, transfer_bundle_path
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso
from .startup_progress import OPTIONAL_STARTUP_STEPS, make_progress, progress_warning
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
    force_update: bool = False,
    skip_steps: set[str] | None = None,
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

    skip_steps = skip_steps or set()

    def emit_progress(name: str, status: str, **kwargs) -> None:
        _notify_progress(progress_callback, make_progress(name, status, **kwargs))

    def run_step(name: str, func: Callable[[ProgressCallback | None], dict[str, object]]) -> dict[str, object]:
        if name in skip_steps or (name in OPTIONAL_STARTUP_STEPS and f"skip:{name}" in skip_steps):
            step = {"name": name, "status": "skipped", "started_utc": utc_now_iso(), "finished_utc": utc_now_iso(), "summary": {"message": "skipped by request"}}
            state["steps"].append(step)
            emit_progress(name, "skipped", progress=1.0, message="skipped by request")
            return dict(step)
        started_utc = utc_now_iso()
        started = monotonic()
        step = {"name": name, "status": "running", "started_utc": started_utc, "finished_utc": "", "summary": {}}
        state["steps"].append(step)
        emit_progress(name, "running", progress=None, message="starting", started_utc=started_utc)

        def subprogress(update: dict[str, object]) -> None:
            elapsed = monotonic() - started
            emit_progress(
                name,
                "running",
                progress=update.get("progress"),
                current=update.get("current"),
                total=update.get("total"),
                message=str(update.get("message", "")),
                started_utc=started_utc,
                elapsed_s=elapsed,
                warning=progress_warning(name, elapsed),
            )

        try:
            summary = func(subprogress)
            step["status"] = str(summary.pop("status", "ok"))
            step["summary"] = summary
        except Exception as exc:
            step["status"] = "failed"
            step["summary"] = {"error": str(exc)}
        step["finished_utc"] = utc_now_iso()
        elapsed = monotonic() - started
        emit_progress(
            name,
            "failed" if step["status"] == "failed" else "skipped" if step["status"] == "skipped" else "done",
            progress=1.0,
            message=str(dict(step["summary"]).get("message") or dict(step["summary"]).get("skipped_reason") or dict(step["summary"]).get("path") or ""),
            started_utc=started_utc,
            finished_utc=str(step["finished_utc"]),
            elapsed_s=elapsed,
            error=str(dict(step["summary"]).get("error")) if step["status"] == "failed" else None,
        )
        return dict(step)

    run_step("init_project", lambda progress: _init_summary(project_name, progress))
    run_step("validate", lambda progress: _validate_summary(project_name, progress))

    if run_training:
        run_step("train_baseline", lambda progress: _baseline_summary(train_and_save_project_baseline(project_name), models, progress))
        run_step("train_classifier", lambda progress: _classifier_summary(train_and_save_project_classifier(project_name), models, progress))
        run_step("train_timeseries", lambda progress: _timeseries_summary(train_and_save_project_timeseries(project_name), models, progress))
    else:
        run_step("load_models", lambda progress: _load_model_summaries(project_name, models, progress))

    run_step("evaluate", lambda progress: _evaluation_summary(evaluate_project_scenes(project_name), models, progress))

    if run_watchers:
        run_step("watcher", lambda progress: _watcher_summary(run_watcher_scan(project_name, duration=0.05, tick_hz=20), project_name, models, progress))
    else:
        run_step("watcher", lambda progress: _watcher_existing_summary(project_name, models, progress))

    if run_orbiters:
        run_step("orbiters", lambda progress: _orbiters_summary(project_name, models, progress, startup=startup, force=force_update))
    else:
        run_step("orbiters", lambda progress: _orbiters_existing_summary(project_name, models, progress))

    if run_transfer:
        run_step("transfer", lambda progress: _transfer_summary(export_transfer_bundle(project_name), models, progress))
    else:
        run_step("transfer", lambda progress: _transfer_existing_summary(project_name, models, progress))

    state["council"] = build_council_summary(project_name, models)
    state["status"] = _overall_status(list(state["steps"]), dict(state["council"]))
    run_step("write_state", lambda progress: _write_state_summary(project_name, state, progress))
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
    grouped_label_counts = _repeatability_label_counts(label_counts)

    if baseline_count < 3:
        warnings.append("not enough baseline scenes")
        recommendations.append("record 3 more baseline takes or run baseline-suite to increase negative controls")
    repeated_short = sorted(label for label, count in grouped_label_counts.items() if count < 2)
    manual_short = [label for label in repeated_short if _label_capture_kind(label) == "manual"]
    auto_short = [label for label in repeated_short if _label_capture_kind(label) == "auto"]
    if manual_short:
        weak_labels.extend(manual_short)
        warnings.append("not enough repeated user labels")
        recommendations.extend(f"record 3 more takes of label family {label}" for label in manual_short[:5])
    if auto_short:
        weak_labels.extend(auto_short)
        warnings.append("not enough repeated automatic control labels")
        recommendations.extend(_auto_scene_recommendation(project_name, label) for label in auto_short[:5])
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

    agreement = _readiness_agreement(baseline_count, classifier_count, timeseries_count, grouped_label_counts)
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


def _init_summary(project_name: str, progress: ProgressCallback | None = None) -> dict[str, object]:
    if progress:
        progress({"progress": 0.2, "message": "creating project directories"})
    root = init_project(project_name)
    if progress:
        progress({"progress": 1.0, "message": str(root)})
    return {"path": str(root)}


def _validate_summary(project_name: str, progress: ProgressCallback | None = None) -> dict[str, object]:
    if progress:
        progress({"progress": None, "message": "checking scene metadata"})
    result = validate_dataset(project_name)
    return {
        "status": "warning" if result.error_count or result.warning_count else "ok",
        "total_scenes": result.total_scenes,
        "valid_scenes": result.valid_scenes,
        "errors": result.error_count,
        "warnings": result.warning_count,
    }


def _baseline_summary(model, models: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    if progress:
        progress({"progress": 1.0, "current": model.scene_count, "total": model.scene_count, "message": f"trained {model.scene_count} baseline scenes"})
    summary = {"scene_count": model.scene_count, "channel_count": len(model.channels), "channels": sorted(model.channels)}
    models["baseline"] = summary
    return summary


def _classifier_summary(model, models: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    if progress:
        progress({"progress": 1.0, "current": model.scene_count, "total": model.scene_count, "message": f"trained {model.scene_count} scenes"})
    summary = {"scene_count": model.scene_count, "baseline_scene_count": model.baseline_scene_count, "label_counts": model.label_counts}
    models["classifier"] = summary
    return summary


def _timeseries_summary(model, models: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    if progress:
        progress({"progress": 1.0, "current": model.scene_count, "total": model.scene_count, "message": f"trained {model.scene_count} temporal profiles"})
    summary = {"scene_count": model.scene_count, "label_counts": model.label_counts, "sequence_channels": model.sequence_channels}
    models["timeseries"] = summary
    return summary


def _load_model_summaries(project_name: str, models: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    if progress:
        progress({"progress": 0.2, "message": "loading existing artifacts"})
    baseline = load_project_baseline(project_name)
    classifier = load_project_classifier(project_name)
    timeseries = load_project_timeseries(project_name)
    if baseline is not None:
        models["baseline"] = {"scene_count": baseline.scene_count, "channel_count": len(baseline.channels), "channels": sorted(baseline.channels)}
    if classifier is not None:
        models["classifier"] = {"scene_count": classifier.scene_count, "baseline_scene_count": classifier.baseline_scene_count, "label_counts": classifier.label_counts}
    if timeseries is not None:
        models["timeseries"] = {"scene_count": timeseries.scene_count, "label_counts": timeseries.label_counts, "sequence_channels": timeseries.sequence_channels}
    if progress:
        progress({"progress": 1.0, "message": "existing artifacts loaded"})
    return {"loaded": True}


def _evaluation_summary(report: dict[str, object], models: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    if progress:
        progress({"progress": 1.0, "current": report.get("scene_count", 0), "total": report.get("scene_count", 0), "message": "evaluation report written"})
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


def _watcher_summary(result: dict[str, object], project_name: str, models: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    events = read_recent_watcher_events(project_name, 10)
    summary = {
        "event_count": len(events),
        "recent_events": events,
        "last_scene_id": dict(result.get("scene", {})).get("scene_id"),
        "path": result.get("watcher_events_path", ""),
    }
    models["watcher"] = summary
    if progress:
        progress({"progress": 1.0, "current": len(events), "total": len(events), "message": "watcher refresh complete"})
    return summary


def _watcher_existing_summary(project_name: str, models: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    events = read_recent_watcher_events(project_name, 10)
    summary = {"status": "skipped", "event_count": len(events), "recent_events": events}
    models["watcher"] = summary
    if progress:
        progress({"progress": 1.0, "message": "skipped: disabled"})
    return summary


def _orbiters_summary(project_name: str, models: dict[str, object], progress: ProgressCallback | None = None, *, startup: bool = False, force: bool = False) -> dict[str, object]:
    if startup:
        report = update_project_orbiters_incremental(project_name, force=force, enrich=False, progress_callback=progress)
        summaries = read_recent_orbiter_summaries(project_path(project_name), 10)
        summary = {"status": report.get("status", "ok"), "processed_events": report.get("processed_events", 0), "summary_count": len(summaries), "recent_summaries": summaries, "message": report.get("skipped_reason") or "incremental orbiters updated"}
        models["orbiters"] = summary
        return summary
    report = evaluate_project_orbiters(project_name)
    summaries = read_recent_orbiter_summaries(project_path(project_name), 10)
    summary = {"evaluated": report.get("evaluated", 0), "accuracy": report.get("accuracy", 0.0), "summary_count": len(summaries), "recent_summaries": summaries}
    models["orbiters"] = summary
    return summary


def _orbiters_existing_summary(project_name: str, models: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    summaries = read_recent_orbiter_summaries(project_path(project_name), 10)
    summary = {"status": "skipped", "summary_count": len(summaries), "recent_summaries": summaries}
    models["orbiters"] = summary
    if progress:
        progress({"progress": 1.0, "message": "skipped: disabled"})
    return summary


def _transfer_summary(bundle: dict[str, object], models: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    summary = {"path": str(transfer_bundle_path(str(bundle.get("project_name", "")))), "total_scenes": bundle.get("total_scenes", 0), "scene_counts": bundle.get("scene_counts", {})}
    models["transfer"] = summary
    if progress:
        progress({"progress": 1.0, "current": bundle.get("total_scenes", 0), "total": bundle.get("total_scenes", 0), "message": "transfer bundle written"})
    return summary


def _transfer_existing_summary(project_name: str, models: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    path = transfer_bundle_path(project_name)
    summary: dict[str, object] = {"status": "skipped", "path": str(path), "exists": path.exists()}
    models["transfer"] = summary
    if progress:
        progress({"progress": 1.0, "message": "skipped: disabled"})
    return summary


def _write_state_summary(project_name: str, state: dict[str, object], progress: ProgressCallback | None = None) -> dict[str, object]:
    path = intelligence_state_path(project_name)
    ensure_dir(path.parent)
    write_json(path, state)
    if progress:
        progress({"progress": 1.0, "message": str(path)})
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


def _repeatability_label_counts(label_counts: dict[str, int]) -> dict[str, int]:
    grouped: dict[str, int] = {}
    for label, count in label_counts.items():
        if _is_baseline_label(label):
            continue
        family = _label_family(label)
        grouped[family] = grouped.get(family, 0) + int(count)
    return grouped


def _label_family(label: str) -> str:
    normalized = _normalize_label(label)
    for prefix in ("user_", "person_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    tokens = [token for token in normalized.split("_") if token]
    if not tokens:
        return "unknown"
    if "approach" in tokens or "approaches" in tokens:
        return "approach"
    if "typing" in tokens or "keyboard" in tokens:
        return "typing"
    if "mouse" in tokens:
        return "mouse_activity"
    if "phone" in tokens:
        return "phone_near"
    if "door" in tokens:
        return "door_open_close"
    if "table" in tokens and "tap" in tokens:
        return "table_tap"
    if "stand" in tokens or "stationary" in tokens:
        return "stand_near"
    if "sit" in tokens:
        return "sit_down"
    if "leave" in tokens or "depart" in tokens or "departs" in tokens:
        return "leave"
    if "walk" in tokens or "walks" in tokens:
        direction_tokens = {"left", "right", "front", "behind", "to"}
        direction = [token for token in tokens if token in direction_tokens]
        return "walk_" + "_".join(direction) if direction else "walk"
    return normalized


def _label_capture_kind(label: str) -> str:
    normalized = _normalize_label(label)
    if normalized.startswith("activity_"):
        return "auto"
    scenario = _scenario_by_normalized_label(normalized)
    if scenario is not None:
        return "auto" if scenario.automatable else "manual"
    return "manual"


def _auto_scene_recommendation(project_name: str, label: str) -> str:
    scenario = _scenario_by_normalized_label(_normalize_label(label))
    include = scenario.label if scenario is not None else label
    return f"run: python -m dsense auto-scenes {project_name} --include {include} --repeat 2 --yes"


def _scenario_by_normalized_label(label: str):
    for scenario in all_scenarios():
        if _normalize_label(scenario.label) == label:
            return scenario
    return None


def _is_baseline_label(label: str) -> bool:
    return _normalize_label(label).startswith("baseline_")


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_") or "unknown"


def _notify_progress(callback: ProgressCallback | None, event: dict[str, object]) -> None:
    if callback is not None:
        callback(event)


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
