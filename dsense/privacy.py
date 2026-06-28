from __future__ import annotations

from pathlib import Path

from .manifest import project_path
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso

SENSITIVE_LABEL_TERMS = {
    "person",
    "user",
    "phone",
    "door",
    "walk",
    "typing",
    "mouse",
    "home",
    "office",
    "bedroom",
}
SENSITIVE_CHANNEL_TERMS = {
    "battery",
    "power",
    "thermal",
    "network",
    "linux_self",
}


def privacy_report_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "privacy_report.json"


def build_privacy_report(project_name: str, out_path: Path | None = None) -> dict[str, object]:
    scenes = _load_scenes(project_name)
    labels = [str(scene.get("label", "")) for scene in scenes]
    notes_count = sum(1 for scene in scenes if str(scene.get("notes", "")).strip())
    timestamp_count = sum(1 for scene in scenes if scene.get("created_utc") or scene.get("labeled_utc"))
    channel_ids = sorted({
        str(channel.get("id", ""))
        for scene in scenes
        for channel in scene.get("channels", [])
        if isinstance(channel, dict)
    })
    sensitive_labels = sorted({
        label for label in labels
        if any(term in label.lower() for term in SENSITIVE_LABEL_TERMS)
    })
    sensitive_channels = sorted({
        channel for channel in channel_ids
        if any(term in channel.lower() for term in SENSITIVE_CHANNEL_TERMS)
    })
    warnings = []
    if sensitive_labels:
        warnings.append("Labels may reveal user behavior, room events, or routines.")
    if notes_count:
        warnings.append("Scene notes may contain free-form private context.")
    if timestamp_count:
        warnings.append("Scene timestamps can reveal routines or recording sessions.")
    if sensitive_channels:
        warnings.append("Some channels may fingerprint power, thermal, network, or process state.")
    if len(scenes) >= 10:
        warnings.append("Repeated scene counts can reveal routine structure.")
    report = {
        "format": "dsense-privacy-report-v1",
        "created_utc": utc_now_iso(),
        "project_name": project_name,
        "scene_count": len(scenes),
        "label_count": len(set(labels)),
        "sensitive_labels": sensitive_labels,
        "sensitive_channels": sensitive_channels,
        "notes_count": notes_count,
        "timestamp_count": timestamp_count,
        "warnings": warnings,
        "recommendations": [
            "Review labels before sharing.",
            "Use export-transfer --redact for model-stat transfer bundles.",
            "Do not share raw scene folders unless every label, note, timestamp, and channel is intentional.",
        ],
    }
    out = out_path or privacy_report_path(project_name)
    ensure_dir(out.parent)
    write_json(out, report)
    return report


def print_privacy_report(report: dict[str, object]) -> None:
    print(f"Privacy report: {report['project_name']}")
    print(f"Scenes: {report['scene_count']}  Labels: {report['label_count']}")
    print(f"Sensitive labels: {', '.join(report.get('sensitive_labels', [])) or 'none'}")
    print(f"Sensitive channels: {', '.join(report.get('sensitive_channels', [])) or 'none'}")
    print(f"Notes-bearing scenes: {report.get('notes_count', 0)}")
    print(f"Timestamped scenes: {report.get('timestamp_count', 0)}")
    warnings = list(report.get("warnings", []))
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print("Recommendations:")
    for recommendation in report.get("recommendations", []):
        print(f"  - {recommendation}")


def redact_transfer_bundle(bundle: dict[str, object]) -> dict[str, object]:
    redacted = dict(bundle)
    redacted["format"] = "dsense-transfer-v1-redacted"
    redacted["redacted"] = True
    redacted["redaction"] = {
        "removed": ["project_name", "created_utc", "label_counts", "label_profiles", "trained_utc"],
        "kept": ["scene_counts", "quality", "baseline channel statistics", "available channel ids"],
    }
    redacted["project_name"] = "redacted"
    redacted.pop("created_utc", None)
    redacted["channels"] = [
        {
            "id": channel.get("id"),
            "group": channel.get("group", "portable"),
            "available": channel.get("available", False),
        }
        for channel in redacted.get("channels", [])
        if isinstance(channel, dict)
    ]
    if isinstance(redacted.get("baseline"), dict):
        redacted["baseline"] = _redact_model(dict(redacted["baseline"]), keep_label_data=False)
    if isinstance(redacted.get("classifier"), dict):
        redacted["classifier"] = _redact_model(dict(redacted["classifier"]), keep_label_data=False)
    redacted["sharing_summary"] = {
        "total_scenes": redacted.get("total_scenes", 0),
        "scene_counts": redacted.get("scene_counts", {}),
        "contains_raw_scenes": False,
        "contains_labels": False,
        "contains_notes": False,
        "contains_timestamps": False,
    }
    return redacted


def _redact_model(model: dict[str, object], keep_label_data: bool) -> dict[str, object]:
    redacted = dict(model)
    redacted["project_name"] = "redacted"
    redacted.pop("trained_utc", None)
    if not keep_label_data:
        redacted.pop("label_counts", None)
        redacted.pop("label_profiles", None)
    return redacted


def _load_scenes(project_name: str) -> list[dict[str, object]]:
    scenes = []
    for path in sorted((project_path(project_name) / "scenes").glob("scene_*/scene.json")):
        try:
            scenes.append(read_json(path))
        except (OSError, ValueError):
            continue
    return scenes
