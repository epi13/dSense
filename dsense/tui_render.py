from __future__ import annotations

import textwrap

from .classifier import SceneClassifierModel


TABS = ["Live", "Sense Radar", "Council", "Capture", "Scenes", "Evaluation", "Watchers", "Orbiters", "Transfer", "Settings"]
MIN_TUI_HEIGHT = 18
MIN_TUI_WIDTH = 60
SPARKLINE_LEVELS = "▁▂▃▄▅▆▇█"
VALUE_CHANNEL_IDS = {
    "dt_ns": "clock_delta",
    "sleep_drift_ns": "sleep_jitter",
    "process_ns_estimate": "process_probe",
    "probe_acc": "process_probe",
    "cpu_load_ppm": "cpu_load",
    "disk_stat_latency_ns": "disk_latency",
    "network_latency_ns": "network_latency",
    "power_online": "power_state",
    "battery_percent": "power_state",
    "linux_ctxt_total": "linux_proc_stat",
    "linux_procs_running": "linux_proc_stat",
    "linux_procs_blocked": "linux_proc_stat",
    "linux_self_vmrss_kb": "linux_proc_self",
    "linux_self_voluntary_ctxt": "linux_proc_self",
    "linux_self_nonvoluntary_ctxt": "linux_proc_self",
    "linux_mem_available_kb": "linux_memory",
    "linux_mem_free_kb": "linux_memory",
    "linux_cpu_temp_millic": "linux_thermal",
}


def summarize_scene_counts(scenes: list[dict[str, object]]) -> dict[str, int]:
    counts = {"baseline": 0, "user": 0, "other": 0}
    for scene in scenes:
        label = str(scene.get("label", ""))
        if label.startswith("baseline_"):
            counts["baseline"] += 1
        elif label.startswith("user_") or label.startswith("person_") or label in {"Approach", "typing_burst", "mouse_activity", "door_open_close", "phone_near_computer"}:
            counts["user"] += 1
        else:
            counts["other"] += 1
    return counts


def wrap_text(text: str, width: int) -> list[str]:
    cleaned = str(text or "").strip() or "(no notes)"
    if width <= 1:
        return [cleaned[:1] or " "]
    lines: list[str] = []
    for paragraph in cleaned.splitlines() or [cleaned]:
        wrapped = textwrap.wrap(paragraph, width=width, break_long_words=True, break_on_hyphens=False) or [""]
        lines.extend(wrapped)
    return lines or ["(no notes)"]


def clip_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    value = str(text)
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def tab_index_delta(index: int, delta: int) -> int:
    if not TABS:
        return 0
    return (index + delta) % len(TABS)


def scene_detail_lines(scene: dict[str, object]) -> list[str]:
    quality = scene.get("quality", {})
    quality = quality if isinstance(quality, dict) else {}
    return [
        f"Scene ID: {scene.get('scene_id', '?')}",
        f"Label: {scene.get('label', '?')}",
        f"Created: {scene.get('created_utc', '?')}",
        f"Duration: {scene.get('duration_ms', '?')} ms | tick {scene.get('tick_hz', '?')} Hz",
        f"Window: pre {scene.get('pre_roll_ms', '?')} ms | action {scene.get('action_start_ms', '?')}-{scene.get('action_end_ms', '?')} ms | post {scene.get('post_roll_ms', '?')} ms",
        f"Accepted: {scene.get('accepted', False)} | events {scene.get('user_event_count', 0)}",
        f"Quality: confidence {quality.get('confidence', '?')} | frames {quality.get('actual_frames', '?')}/{quality.get('expected_frames', '?')}",
        f"Checksum: {quality.get('checksum_ok', '?')} | frame size {quality.get('frame_size_valid', '?')}",
        f"Availability mask: {quality.get('channel_availability_mask', '?')}",
    ]


def profile_line(channel: str, profile: dict[str, float]) -> str:
    return (
        f"{channel:<30} "
        f"{float(profile.get('center', 0.0)):>12.3g} "
        f"{float(profile.get('mad', 0.0)):>12.3g} "
        f"{float(profile.get('p95', 0.0)):>12.3g} "
        f"{float(profile.get('p99', 0.0)):>12.3g}"
    )


def classifier_summary_lines(model: SceneClassifierModel | None, auto_detect: bool) -> list[str]:
    if model is None:
        return [
            "No classifier trained yet",
            "Accepted scenes will train it",
            "Press u to update intelligence",
        ]

    label_count = len(model.label_counts)
    channels = ", ".join(sorted(model.detector_baseline)) or "none"
    auto_text = "using learned baseline" if auto_detect and model.detector_baseline else "not used by auto events"
    top_labels = sorted(model.label_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    label_text = ", ".join(f"{label}:{count}" for label, count in top_labels) or "none"
    trained = model.trained_utc.replace("T", " ").replace("Z", " UTC")
    if "." in trained:
        trained = trained.split(".", 1)[0] + " UTC"
    return [
        "Active",
        f"trained scenes {model.scene_count} | baseline {model.baseline_scene_count}",
        f"labels {label_count} | {auto_text}",
        f"channels {channels}",
        f"top {label_text}",
        f"trained {trained}",
    ]


def evaluation_repeatability_lines(report: dict[str, object] | None) -> list[str]:
    if not report:
        return ["not evaluated yet"]
    within = dict(report.get("within_label_similarity", {}))
    between = dict(report.get("between_label_distance", {}))
    confusion = dict(report.get("confusion_matrix", {}))
    drift = dict(report.get("baseline_drift", {}))
    return [
        f"within-label similarity: {_fmt_float(within.get('overall'), 2)}",
        f"between-label distance: {_fmt_float(between.get('average'), 2)}",
        f"leave-one-out accuracy: {_fmt_percent(confusion.get('accuracy'))}",
        f"baseline drift max: {_fmt_float(drift.get('max_drift'), 2)}",
    ]


def labels_needing_more_takes(report: dict[str, object] | None, limit: int = 8) -> list[tuple[str, str]]:
    if not report:
        return [("none", "run evaluation")]
    label_counts = {str(label): int(count) for label, count in dict(report.get("label_counts", {})).items()}
    within = {str(label): float(score) for label, score in dict(dict(report.get("within_label_similarity", {})).get("labels", {})).items()}
    matrix = {
        str(actual): {str(predicted): int(count) for predicted, count in dict(predictions).items()}
        for actual, predictions in dict(dict(report.get("confusion_matrix", {})).get("matrix", {})).items()
    }
    rows: list[tuple[int, str, str]] = []
    for label, count in sorted(label_counts.items()):
        if count < 2:
            rows.append((0, label, f"{count} take" if count == 1 else f"{count} takes"))
            continue
        predictions = matrix.get(label, {})
        misses = {predicted: value for predicted, value in predictions.items() if predicted != label and value > 0}
        if misses:
            confused_with, miss_count = max(misses.items(), key=lambda item: (item[1], item[0]))
            rows.append((1, label, f"confused with {confused_with} ({miss_count})"))
            continue
        if float(within.get(label, 1.0)) < 0.7:
            rows.append((2, label, "high variance"))
    if not rows:
        return [("none", "no weak labels detected")]
    return [(label, reason) for _, label, reason in sorted(rows)[:limit]]


def useful_channel_lines(report: dict[str, object] | None, limit: int = 8) -> list[str]:
    if not report:
        return ["none"]
    ranking = report.get("channel_usefulness_ranking", [])
    if not isinstance(ranking, list) or not ranking:
        return ["none"]
    lines = []
    for item in ranking[:limit]:
        row = dict(item)
        lines.append(f"{str(row.get('channel', 'unknown')):<26} score {_fmt_float(row.get('score'), 2)}")
    return lines or ["none"]


def council_summary_lines(state: dict[str, object] | None, limit: int = 8) -> list[str]:
    if not state:
        return ["Status: not updated", "Press u to update the local intelligence stack."]
    council = dict(state.get("council", {}))
    models = dict(state.get("models", {}))
    baseline = dict(models.get("baseline", {}))
    classifier = dict(models.get("classifier", {}))
    timeseries = dict(models.get("timeseries", {}))
    watcher = dict(models.get("watcher", {}))
    orbiters = dict(models.get("orbiters", {}))
    lines = [
        f"Status: {state.get('status', 'unknown')} | confidence {_fmt_float(council.get('overall_confidence'), 2)} | agreement {council.get('agreement', 'unknown')}",
        f"Baseline: {baseline.get('scene_count', 0)} scenes | {baseline.get('channel_count', len(list(baseline.get('channels', []))))} channels",
        f"Classifier: {classifier.get('scene_count', 0)} scenes | {len(dict(classifier.get('label_counts', {})))} labels",
        f"Time-series: {timeseries.get('scene_count', 0)} scenes | {len(list(timeseries.get('sequence_channels', [])))} sequence channels",
        f"Watcher: {watcher.get('event_count', 0)} recent events",
        f"Orbiters: {orbiters.get('summary_count', 0)} summaries",
    ]
    best = list(council.get("best_channels", []))
    if best:
        lines.append("Best channels: " + ", ".join(str(item) for item in best[:5]))
    warnings = [str(item) for item in list(council.get("warnings", []))[:limit]]
    recommendations = [str(item) for item in list(council.get("recommendations", []))[:limit]]
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {item}" for item in warnings)
    if recommendations:
        lines.append("Recommendations:")
        lines.extend(f"  - {item}" for item in recommendations)
    return lines


LIVE_FOOTER = "m mark | r record | u update intelligence | s snapshot | tab tabs | q quit"


def live_footer_text() -> str:
    return LIVE_FOOTER


def live_observation_lines(observation: object | None, width: int = 80) -> list[str]:
    if observation is None:
        return [clip_text(line, width) for line in ["LIVE OBSERVATORY", "waiting for local telemetry...", LIVE_FOOTER]]
    data = observation.to_dict() if hasattr(observation, "to_dict") else dict(observation)
    values = dict(data.get("channel_values", {}))
    status = dict(data.get("channel_status", {}))
    known = list(data.get("known_anomalies", []))
    unknown = list(data.get("unknown_anomalies", []))
    proximity = dict(data.get("proximity_hypothesis", {}))
    rate = _rate_from_elapsed(data)
    lines = [
        f"dSense LIVE OBSERVATORY - project: {data.get('project_name', '?')}       tick: {int(data.get('tick', 0)):06d}       rate: {rate}",
        "",
        "Telemetry",
        _telemetry_strip(values, status, width),
        "",
        "Sense Radar",
        *sense_radar_lines(proximity, data.get("council_agreement", "unknown"), width),
        "",
        "Intelligence Council",
        compact_live_council_line(data, width),
        "",
        "Known Anomalies",
        *anomaly_lines(known, none_text="none", width=width),
        "",
        "Unknown Anomalies",
        *anomaly_lines(unknown, none_text="none", width=width),
        "",
        LIVE_FOOTER,
    ]
    return [clip_text(line, width) for line in lines]


def compact_live_observation_lines(observation: object | None, width: int = 80) -> list[str]:
    if observation is None:
        return ["LIVE OBSERVATORY", "waiting for telemetry", LIVE_FOOTER]
    data = observation.to_dict() if hasattr(observation, "to_dict") else dict(observation)
    unknown = list(data.get("unknown_anomalies", []))
    known = list(data.get("known_anomalies", []))
    top = unknown[:1] or known[:1]
    anomaly = "none" if not top else f"{dict(top[0]).get('name', 'pattern')} score {_fmt_float(dict(top[0]).get('score'), 2)}"
    return [
        f"LIVE OBSERVATORY {data.get('project_name', '?')} tick {int(data.get('tick', 0)):06d}",
        compact_live_council_line(data, width),
        f"Top anomaly: {anomaly}",
        LIVE_FOOTER,
    ]


def compact_live_council_line(data: dict[str, object], width: int = 80) -> str:
    baseline_score = _fmt_float(data.get("baseline_score"), 2)
    watcher_score = _fmt_float(data.get("watcher_score"), 2)
    classifier_label = data.get("classifier_label", "unknown")
    classifier_conf = _fmt_float(data.get("classifier_confidence"), 2)
    timeseries_label = data.get("timeseries_label", "unknown")
    timeseries_conf = _fmt_float(data.get("timeseries_confidence"), 2)
    agreement = data.get("council_agreement", "unknown")
    council_conf = _fmt_float(data.get("council_confidence"), 2)
    return clip_text(
        f"baseline: {baseline_score} | classifier: {classifier_label} {classifier_conf} | "
        f"timeseries: {timeseries_label} {timeseries_conf} | watcher: {watcher_score} | "
        f"agreement: {agreement} | confidence: {council_conf}",
        width,
    )


def sense_radar_lines(proximity: dict[str, object], agreement: object = "unknown", width: int = 80) -> list[str]:
    strength = _fmt_float(proximity.get("strength"), 2)
    direction = str(proximity.get("direction", "unknown"))
    status = str(proximity.get("status", "normal"))
    label_hint = proximity.get("label_hint") or "none"
    confidence = _fmt_float(proximity.get("confidence"), 2)
    radar = [
        "                 N",
        "             .   |   .",
        "        W -------+------- E",
        "             .   |   .",
        "                 S",
        f"strength: {strength}   direction: {direction}   status: {status}",
        f"known match: {label_hint}   confidence: {confidence}",
        f"council: {agreement}   status: needs validation",
    ]
    return [clip_text(line, width) for line in radar]


def anomaly_lines(rows: list[object], none_text: str, width: int = 80) -> list[str]:
    if not rows:
        return [none_text]
    lines = []
    for row in rows[:5]:
        item = dict(row)
        action = f"   action: {item.get('action')}" if item.get("action") else ""
        lines.append(clip_text(f"{item.get('name', 'pattern'):<24} score {_fmt_float(item.get('score'), 2)}   {item.get('detail', '')}{action}", width))
    return lines


def _telemetry_strip(values: dict[str, object], status: dict[str, object], width: int) -> str:
    if not values:
        return "no live numeric channels yet"
    parts = []
    for key in sorted(values)[:8]:
        label = value_channel_id(key)
        parts.append(f"{label}: {status.get(key, 'ok')}")
    return clip_text("   ".join(parts), width)


def _rate_from_elapsed(data: dict[str, object]) -> str:
    tick = int(data.get("tick", 0) or 0)
    elapsed = float(data.get("elapsed_s", 0.0) or 0.0)
    if elapsed <= 0:
        return "0.0 Hz"
    return f"{tick / elapsed:0.1f} Hz"


def _fmt_float(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.00"


def _fmt_percent(value: object) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "0%"


def format_metric_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def channel_state_label(channel: dict[str, object]) -> str:
    if channel.get("unavailable") or not channel.get("available", False):
        return "offline"
    if channel.get("sampled"):
        return "sampled"
    if channel.get("stale"):
        return "stale"
    return "idle"


def value_channel_id(value_name: str) -> str:
    return VALUE_CHANNEL_IDS.get(value_name, value_name)


def robust_channel_score(name: str, value: object, baseline_profiles: dict[str, dict[str, float]] | None, history: list[float] | None = None) -> float:
    numeric = _float_or_none(value)
    if numeric is None:
        return 0.0
    profile = (baseline_profiles or {}).get(name)
    if profile:
        center = float(profile.get("center", 0.0) or 0.0)
        mad = float(profile.get("mad", 1.0) or 1.0)
        return abs(numeric - center) / mad
    values = [float(item) for item in (history or []) if isinstance(item, (int, float))]
    if len(values) < 4:
        return 0.0
    center = sorted(values)[len(values) // 2]
    deviations = sorted(abs(item - center) for item in values)
    mad = deviations[len(deviations) // 2] or 1.0
    return abs(numeric - center) / mad


def sparkline(values: list[float] | tuple[float, ...], width: int = 10) -> str:
    if width <= 0:
        return ""
    if not values:
        return "-" * width
    tail = [float(value) for value in values[-width:]]
    if len(tail) < width:
        tail = [tail[0]] * (width - len(tail)) + tail
    low = min(tail)
    high = max(tail)
    if high == low:
        return "─" * width
    span = high - low
    chars = []
    for value in tail:
        index = int(round(((value - low) / span) * (len(SPARKLINE_LEVELS) - 1)))
        chars.append(SPARKLINE_LEVELS[max(0, min(len(SPARKLINE_LEVELS) - 1, index))])
    return "".join(chars)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def scheduled_scene_events(duration: float, pre_roll: float, action: float, post_roll: float) -> list[dict[str, object]]:
    action_start_ms = int(pre_roll * 1000)
    action_end_ms = int((pre_roll + action) * 1000)
    return [
        {"t_ms": 0, "event": "scene_start", "source": "system"},
        {"t_ms": action_start_ms, "event": "action_start", "source": "system"},
        {"t_ms": action_end_ms, "event": "action_end", "source": "system"},
        {"t_ms": int(duration * 1000), "event": "scene_end", "source": "system"},
    ]


def system_event_marker(event_name: str) -> str:
    return {
        "scene_start": "S",
        "action_start": "A",
        "action_end": "E",
        "scene_end": "X",
    }.get(event_name, "?")
