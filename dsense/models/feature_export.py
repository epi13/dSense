from __future__ import annotations

from pathlib import Path

from dsense.manifest import project_path
from dsense.models.evaluation import evaluate_project_scenes
from dsense.models.features import feature_manifest, read_numeric_preview_rows, summarize_rows
from dsense.utils.files import ensure_dir, read_json, write_json
from dsense.utils.timebase import utc_now_iso


def features_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "features.json"


def extract_project_features(project_name: str, out_path: Path | None = None) -> dict[str, object]:
    root = project_path(project_name)
    scenes = []
    all_features: list[dict[str, float]] = []
    all_rows: list[dict[str, float]] = []
    for scene_path in sorted((root / "scenes").glob("scene_*/scene.json")):
        try:
            scene = read_json(scene_path)
        except (OSError, ValueError):
            continue
        if scene.get("accepted") is False:
            continue
        preview_path = scene_path.parent / "preview.csv"
        if not preview_path.exists():
            continue
        rows = read_numeric_preview_rows(preview_path)
        if not rows:
            continue
        features = summarize_rows(rows)
        all_rows.extend(rows)
        all_features.append(features)
        scenes.append({
            "scene_id": str(scene.get("scene_id", scene_path.parent.name)),
            "label": str(scene.get("label", "unknown")),
            "preview_csv": str(preview_path),
            "channels": sorted({column for row in rows for column in row}),
            "features": features,
        })
    report = {
        "format": "dsense-features-v1",
        "project_name": project_name,
        "created_utc": utc_now_iso(),
        "scene_count": len(scenes),
        "feature_manifest": feature_manifest(all_features, all_rows),
        "scenes": scenes,
    }
    out = out_path or features_path(project_name)
    ensure_dir(out.parent)
    write_json(out, report)
    return report


def rank_project_channels(project_name: str) -> list[dict[str, object]]:
    report = evaluate_project_scenes(project_name)
    ranking = report.get("channel_usefulness_ranking", [])
    return list(ranking) if isinstance(ranking, list) else []
