from __future__ import annotations

import json
from pathlib import Path

from .gemma_edge import enrich_orbiter_summary
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
    summary = f"{status} substrate window; strongest channel {channel}; classifier suggests {label} ({confidence})."
    return {
        "created_utc": utc_now_iso(),
        "scene_id": scene_id,
        "baseline_status": baseline_status,
        "classifier_prediction": classifier_prediction,
        "anomaly_score": anomaly_score,
        "channel_availability_mask": channel_availability,
        "quality_flags": quality_flags,
        "summary": summary,
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
