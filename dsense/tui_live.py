from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .baseline import BaselineModel, score_against_baseline
from .channels import default_channels
from .classifier import SceneClassifierModel, predict_features
from .manifest import project_path
from .models.evaluation import predict_from_profiles
from .models.features import summarize_rows, variance
from .recorder import _prepare_channel_runtimes, _sample_runtimes, _stop_channel_runtimes
from .timeseries import TimeSeriesModel, extract_timeseries_features
from .utils.files import ensure_dir, write_json
from .utils.timebase import monotonic_ns, utc_now_iso


@dataclass
class LiveObservation:
    created_utc: str
    project_name: str
    tick: int
    elapsed_s: float
    channel_values: dict[str, float]
    channel_status: dict[str, str]
    baseline_score: float | None
    classifier_label: str | None
    classifier_confidence: float | None
    timeseries_label: str | None
    timeseries_confidence: float | None
    watcher_score: float | None
    council_confidence: float | None
    council_agreement: str
    known_anomalies: list[dict[str, object]] = field(default_factory=list)
    unknown_anomalies: list[dict[str, object]] = field(default_factory=list)
    proximity_hypothesis: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    interval_classification: str = "normal"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LiveSampler:
    def __init__(self, channel_groups: list[str] | tuple[str, ...], tick_hz: int):
        self.tick_hz = max(1, int(tick_hz))
        self.interval_ns = int(1_000_000_000 / self.tick_hz)
        self.runtimes = _prepare_channel_runtimes(default_channels(channel_groups), self.tick_hz)
        self.start_ns = monotonic_ns()
        self.tick = 0

    def sample(self) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
        target_ns = self.start_ns + self.tick * self.interval_ns
        sleep_s = (target_ns - monotonic_ns()) / 1_000_000_000
        if sleep_s > 0:
            time.sleep(sleep_s)
        now_ns = monotonic_ns()
        values, availability, quality, sampled, stale, unavailable = _sample_runtimes(self.runtimes, self.tick, now_ns, target_ns)
        status = channel_status_from_masks(values, sampled, stale, unavailable)
        masks = {"availability": availability, "quality": quality, "sampled": sampled, "stale": stale, "unavailable": unavailable}
        self.tick += 1
        return {key: float(value) for key, value in values.items()}, status, masks

    def close(self) -> None:
        _stop_channel_runtimes(self.runtimes)


class LiveSessionWriter:
    def __init__(self, project_name: str, min_interval_s: float = 1.0):
        self.project_name = project_name
        self.min_interval_s = min_interval_s
        self.last_write_s = 0.0
        self.last_anomaly_count = 0

    def maybe_write(self, observation: LiveObservation, *, force: bool = False, event: str | None = None) -> bool:
        anomaly_count = len(observation.known_anomalies) + len(observation.unknown_anomalies)
        now = time.monotonic()
        should_write = force or event is not None or anomaly_count > self.last_anomaly_count or now - self.last_write_s >= self.min_interval_s
        if not should_write:
            return False
        append_live_session_event(self.project_name, observation, event=event)
        self.last_write_s = now
        self.last_anomaly_count = anomaly_count
        return True


def channel_status_from_masks(values: dict[str, object], sampled_mask: int, stale_mask: int, unavailable_mask: int) -> dict[str, str]:
    status: dict[str, str] = {}
    for key in values:
        status[key] = "sampled"
    if sampled_mask == 0 and stale_mask == 0 and unavailable_mask == 0:
        return {key: "ok" for key in values}
    for key in values:
        status[key] = "ok"
    return status


def build_live_observation(
    project_name: str,
    *,
    tick: int,
    elapsed_s: float,
    channel_values: dict[str, float],
    channel_status: dict[str, str],
    recent_rows: list[dict[str, float]],
    baseline: BaselineModel | None,
    classifier: SceneClassifierModel | None,
    timeseries: TimeSeriesModel | None,
    council_state: dict[str, object] | None,
    watcher_events: list[dict[str, object]] | None = None,
) -> LiveObservation:
    baseline_status = score_against_baseline(channel_values, baseline)
    baseline_score = float(baseline_status.get("score", 0.0) or 0.0)
    classifier_prediction = predict_features(classifier, summarize_rows(recent_rows)) if recent_rows else {"label": "unknown", "confidence": 0.0}
    timeseries_prediction = _predict_timeseries_rows(timeseries, recent_rows)
    watcher_score = _watcher_score(watcher_events or [], baseline_score)
    classifier_confidence = _bounded_float(classifier_prediction.get("confidence"))
    timeseries_confidence = _bounded_float(timeseries_prediction.get("confidence"))
    classifier_label = str(classifier_prediction.get("label", "unknown"))
    timeseries_label = str(timeseries_prediction.get("label", "unknown"))
    council = dict((council_state or {}).get("council", {})) if isinstance(council_state, dict) else {}
    council_agreement = live_council_agreement(
        baseline_status=str(baseline_status.get("status", "unknown")),
        classifier_label=classifier_label,
        classifier_confidence=classifier_confidence,
        timeseries_label=timeseries_label,
        timeseries_confidence=timeseries_confidence,
        watcher_score=watcher_score,
    )
    unknown_score = score_unknown_anomaly(
        baseline_score=baseline_score,
        watcher_score=watcher_score,
        classifier_confidence=classifier_confidence,
        timeseries_confidence=timeseries_confidence,
        channel_volatility=channel_volatility(recent_rows),
        council_agreement=council_agreement,
    )
    known = known_anomalies_from_observation(
        baseline_status=dict(baseline_status),
        classifier_label=classifier_label,
        classifier_confidence=classifier_confidence,
        watcher_events=watcher_events or [],
    )
    unknown = unknown_anomalies_from_score(
        unknown_score,
        classifier_confidence=classifier_confidence,
        timeseries_confidence=timeseries_confidence,
        council_agreement=council_agreement,
    )
    label_counts = {}
    if classifier is not None:
        label_counts.update(classifier.label_counts)
    if timeseries is not None:
        label_counts.update(timeseries.label_counts)
    proximity = direction_hypothesis_from_labels(sorted(label_counts), classifier_label if classifier_label != "unknown" else timeseries_label)
    proximity["strength"] = round(max(unknown_score, min(baseline_score / 12.0, 1.0)), 3)
    if proximity.get("direction") == "unknown":
        proximity["status"] = "normal" if unknown_score < 0.45 else "possible unknown anomaly"
    else:
        proximity["status"] = "possible proximity pattern" if unknown_score >= 0.35 else "weak unvalidated hypothesis"
    warnings = []
    if council_agreement == "low":
        warnings.append("council disagreement needs validation")
    if unknown_score >= 0.65:
        warnings.append("unknown anomaly score elevated")
    interval = classify_live_interval(known, unknown, unknown_score, council_agreement)
    return LiveObservation(
        created_utc=utc_now_iso(),
        project_name=project_name,
        tick=tick,
        elapsed_s=round(elapsed_s, 3),
        channel_values=channel_values,
        channel_status=channel_status,
        baseline_score=round(baseline_score, 3),
        classifier_label=classifier_label,
        classifier_confidence=classifier_confidence,
        timeseries_label=timeseries_label,
        timeseries_confidence=timeseries_confidence,
        watcher_score=round(watcher_score, 3),
        council_confidence=_bounded_float(council.get("overall_confidence")),
        council_agreement=council_agreement,
        known_anomalies=known,
        unknown_anomalies=unknown,
        proximity_hypothesis=proximity,
        warnings=warnings,
        interval_classification=interval,
    )


def live_council_agreement(
    *,
    baseline_status: str,
    classifier_label: str | None,
    classifier_confidence: float | None,
    timeseries_label: str | None,
    timeseries_confidence: float | None,
    watcher_score: float | None,
) -> str:
    if not classifier_label or not timeseries_label or classifier_label == "unknown" or timeseries_label == "unknown":
        return "unknown"
    watcher_quiet = float(watcher_score or 0.0) < 0.45
    labels_match = classifier_label == timeseries_label
    confident = float(classifier_confidence or 0.0) >= 0.45 and float(timeseries_confidence or 0.0) >= 0.45
    baseline_normal = baseline_status in {"normal", "no_overlap", "untrained"}
    if labels_match and confident and watcher_quiet and baseline_normal:
        return "high"
    if labels_match and confident:
        return "medium"
    return "low"


def score_unknown_anomaly(
    *,
    baseline_score: float | None,
    watcher_score: float | None,
    classifier_confidence: float | None,
    timeseries_confidence: float | None,
    channel_volatility: float,
    council_agreement: str,
) -> float:
    # Transparent heuristic only: blends local distance, low confidence, volatility, and disagreement.
    baseline_component = min(float(baseline_score or 0.0) / 12.0, 1.0)
    watcher_component = min(float(watcher_score or 0.0), 1.0)
    classifier_component = 1.0 - float(classifier_confidence or 0.0)
    timeseries_component = 1.0 - float(timeseries_confidence or 0.0)
    volatility_component = min(max(channel_volatility, 0.0), 1.0)
    disagreement_component = {"high": 0.0, "medium": 0.25, "low": 0.75, "unknown": 0.4}.get(council_agreement, 0.4)
    score = (
        baseline_component * 0.22
        + watcher_component * 0.18
        + classifier_component * 0.16
        + timeseries_component * 0.16
        + volatility_component * 0.12
        + disagreement_component * 0.16
    )
    return round(max(0.0, min(1.0, score)), 3)


def classify_live_interval(
    known_anomalies: list[dict[str, object]],
    unknown_anomalies: list[dict[str, object]],
    unknown_score: float,
    council_agreement: str,
) -> str:
    if unknown_anomalies and (unknown_score >= 0.72 or council_agreement == "low"):
        return "needs_recording"
    if unknown_anomalies:
        return "unknown_anomaly"
    if known_anomalies:
        return "known_anomaly"
    return "normal"


def known_anomalies_from_observation(
    *,
    baseline_status: dict[str, object],
    classifier_label: str,
    classifier_confidence: float | None,
    watcher_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if baseline_status.get("status") == "anomaly":
        channel = str(baseline_status.get("channel", "unknown"))
        name = "timing_spike" if channel in {"dt_ns", "sleep_drift_ns", "process_ns_estimate"} else f"{channel}_disturbance"
        rows.append({"name": name, "score": _rounded_score(float(baseline_status.get("score", 0.0) or 0.0) / 12.0), "detail": "likely system jitter"})
    if classifier_label not in {"unknown", "baseline_idle"} and float(classifier_confidence or 0.0) >= 0.45:
        suffix = "" if float(classifier_confidence or 0.0) >= 0.65 else "?"
        rows.append({"name": classifier_label + suffix, "score": _rounded_score(classifier_confidence), "detail": "trained label match needs validation"})
    for event in watcher_events[-3:]:
        event_name = str(event.get("event", "watcher_event"))
        if event_name:
            rows.append({"name": event_name, "score": _rounded_score(event.get("anomaly_score", 0.0)), "detail": "known watcher event type"})
    return rows[-5:]


def unknown_anomalies_from_score(
    score: float,
    *,
    classifier_confidence: float | None,
    timeseries_confidence: float | None,
    council_agreement: str,
) -> list[dict[str, object]]:
    if score < 0.52:
        return []
    if council_agreement == "low":
        reason = "council disagreement"
        action = "record scene"
    elif float(classifier_confidence or 0.0) < 0.35 or float(timeseries_confidence or 0.0) < 0.35:
        reason = "low model confidence"
        action = "needs repeated sample"
    else:
        reason = "new signal shape"
        action = "mark interval"
    return [{"name": "unclassified pattern", "score": round(score, 3), "detail": reason, "action": action}]


def direction_hypothesis_from_labels(labels: list[str], label_hint: str | None = None) -> dict[str, object]:
    candidates = [label_hint or "", *labels]
    for label in candidates:
        parsed = _direction_from_label(str(label))
        if parsed is not None:
            direction, detail = parsed
            return {
                "status": "weak unvalidated hypothesis",
                "direction": direction,
                "strength": 0.0,
                "label_hint": label,
                "confidence": 0.35,
                "detail": detail,
            }
    return {
        "status": "normal",
        "direction": "unknown",
        "strength": 0.0,
        "label_hint": None,
        "confidence": 0.0,
        "detail": "direction unknown - no trained directional scenes",
    }


def channel_volatility(rows: list[dict[str, float]]) -> float:
    if len(rows) < 4:
        return 0.0
    scores = []
    for channel in sorted({key for row in rows for key in row}):
        values = [float(row.get(channel, 0.0)) for row in rows[-12:]]
        if not values:
            continue
        center = sum(abs(value) for value in values) / len(values)
        scores.append(min(variance(values) / max(center * center, 1.0), 1.0))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def live_session_path(project_name: str) -> Path:
    return project_path(project_name) / "events" / "live_session.jsonl"


def append_live_session_event(project_name: str, observation: LiveObservation, event: str | None = None) -> Path:
    path = live_session_path(project_name)
    ensure_dir(path.parent)
    payload = observation.to_dict()
    if event:
        payload["event"] = event
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def save_live_snapshot(project_name: str, observation: LiveObservation) -> Path:
    stamp = observation.created_utc.replace(":", "").replace("-", "").replace(".", "").replace("Z", "Z")
    path = project_path(project_name) / "exports" / f"live_snapshot_{stamp}.json"
    ensure_dir(path.parent)
    write_json(path, observation.to_dict())
    return path


def _predict_timeseries_rows(model: TimeSeriesModel | None, rows: list[dict[str, float]]) -> dict[str, object]:
    if model is None or not rows or not model.label_profiles:
        return {"label": "unknown", "confidence": 0.0}
    features = extract_timeseries_features(rows)
    return predict_from_profiles(model.label_profiles, features)


def _watcher_score(events: list[dict[str, object]], baseline_score: float) -> float:
    recent = max([float(event.get("anomaly_score", 0.0) or 0.0) for event in events[-5:]] or [0.0])
    return max(min(recent / 12.0, 1.0), min(baseline_score / 12.0, 1.0))


def _direction_from_label(label: str) -> tuple[str, str] | None:
    lowered = label.lower()
    if "left_to_right" in lowered or ("left" in lowered and "right" in lowered and lowered.index("left") < lowered.index("right")):
        return "weak east/right movement hypothesis", "derived from directional trained label"
    if "right_to_left" in lowered or ("right" in lowered and "left" in lowered and lowered.index("right") < lowered.index("left")):
        return "weak west/left movement hypothesis", "derived from directional trained label"
    if "approach" in lowered or "toward" in lowered or "front" in lowered:
        return "weak front/toward hypothesis", "derived from directional trained label"
    if "away" in lowered or "back" in lowered:
        return "weak away/back hypothesis", "derived from directional trained label"
    for word in ("north", "south", "east", "west"):
        if word in lowered:
            return f"weak {word} hypothesis", "derived from directional trained label"
    return None


def _bounded_float(value: object) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError):
        return 0.0


def _rounded_score(value: object) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError):
        return 0.0
