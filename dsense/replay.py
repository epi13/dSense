from __future__ import annotations

from pathlib import Path

from .baseline import load_project_baseline
from .classifier import load_project_classifier, predict_scene
from .event_detector import HeuristicEventDetector
from .frame import FRAME_SIZE, frame_to_dict
from .manifest import DEFAULT_PROJECT, project_path
from .models.features import summarize_preview
from .utils.files import ensure_dir, read_json, write_json


def resolve_scene_dir(target: str, scene_id: str | None = None) -> Path:
    if scene_id is not None:
        return project_path(target) / "scenes" / scene_id
    path = Path(target)
    if path.exists():
        return path
    return project_path(DEFAULT_PROJECT) / "scenes" / target


def inspect_scene(scene_dir: Path) -> dict[str, object]:
    scene = _read_scene(scene_dir)
    frames_path = scene_dir / "frames.ds64"
    preview_path = scene_dir / "preview.csv"
    events = read_events(scene_dir)
    notes_path = scene_dir / "notes.txt"
    frame_count = frames_path.stat().st_size // FRAME_SIZE if frames_path.exists() else 0
    preview_rows = read_preview_rows(preview_path)
    return {
        "scene_id": scene.get("scene_id", scene_dir.name),
        "label": scene.get("label", "unknown"),
        "accepted": scene.get("accepted", False),
        "mode": scene.get("mode", "unknown"),
        "duration_ms": scene.get("duration_ms", 0),
        "tick_hz": scene.get("tick_hz", 0),
        "quality": scene.get("quality", {}),
        "frame_count": frame_count,
        "preview_rows": len(preview_rows),
        "event_count": len(events),
        "events": events,
        "notes": notes_path.read_text(encoding="utf-8").strip() if notes_path.exists() else "",
        "channels": scene.get("channels", []),
        "scene_dir": str(scene_dir),
    }


def inspect_frame(scene_dir: Path, tick: int) -> dict[str, object]:
    frames_path = scene_dir / "frames.ds64"
    if not frames_path.exists():
        raise FileNotFoundError(f"Missing frames.ds64 in {scene_dir}")
    offset = tick * FRAME_SIZE
    with frames_path.open("rb") as handle:
        handle.seek(offset)
        frame = handle.read(FRAME_SIZE)
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"Tick {tick} is outside {frames_path}")
    result: dict[str, object] = {"frame": frame_to_dict(frame)}
    preview_row = preview_row_by_tick(scene_dir / "preview.csv", tick)
    if preview_row is not None:
        result["preview"] = preview_row
    return result


def classify_existing_scene(project_name: str, scene_dir: Path) -> dict[str, object]:
    preview_path = scene_dir / "preview.csv"
    if not preview_path.exists():
        raise FileNotFoundError(f"Missing preview.csv in {scene_dir}")
    model = load_project_classifier(project_name)
    return predict_scene(model, preview_path)


def replay_scene(project_name: str, scene_dir: Path, limit: int = 10) -> dict[str, object]:
    scene = _read_scene(scene_dir)
    preview_path = scene_dir / "preview.csv"
    if not preview_path.exists():
        raise FileNotFoundError(f"Missing preview.csv in {scene_dir}")
    rows = read_preview_rows(preview_path)
    tick_hz = int(scene.get("tick_hz", 100) or 100)
    baseline = load_project_baseline(project_name)
    learned = baseline.channels if baseline is not None else None
    detector = HeuristicEventDetector(tick_hz=tick_hz, learned_baseline=learned)
    detector_events = []
    first_t_ns = _int_value(rows[0].get("t_ns"), 0) if rows else 0
    for idx, row in enumerate(rows):
        t_ns = _int_value(row.get("t_ns"), first_t_ns)
        progress = {
            "tick": _int_value(row.get("tick"), idx),
            "elapsed_ms": int((t_ns - first_t_ns) / 1_000_000) if first_t_ns else int(idx * 1000 / max(tick_hz, 1)),
            "dt_ns": _float_value(row.get("dt_ns"), 0.0),
            "sleep_drift_ns": _float_value(row.get("sleep_drift_ns"), 0.0),
            "process_ns_estimate": _float_value(row.get("process_ns_estimate"), 0.0),
        }
        values = {
            key: _float_value(value, 0.0)
            for key, value in row.items()
            if key not in {"tick", "t_ns"} and _is_numeric(value)
        }
        progress["values"] = values
        progress.update(values)
        detector_events.extend(detector.update(progress))
    return {
        "scene": inspect_scene(scene_dir),
        "classifier_prediction": classify_existing_scene(project_name, scene_dir),
        "recorded_events": read_events(scene_dir),
        "detector_events": detector_events,
        "detector_state": detector.state.__dict__,
        "preview_sample": rows[:limit],
    }


def export_scene_json(project_name: str, scene_dir: Path, out_path: Path | None = None) -> dict[str, object]:
    frames = []
    frames_path = scene_dir / "frames.ds64"
    if frames_path.exists():
        with frames_path.open("rb") as handle:
            while True:
                frame = handle.read(FRAME_SIZE)
                if not frame:
                    break
                if len(frame) == FRAME_SIZE:
                    frames.append(frame_to_dict(frame))
    preview_path = scene_dir / "preview.csv"
    report = {
        "format": "dsense-scene-debug-v1",
        "project_name": project_name,
        "summary": inspect_scene(scene_dir),
        "scene": _read_scene(scene_dir),
        "events": read_events(scene_dir),
        "preview": read_preview_rows(preview_path),
        "features": summarize_preview(preview_path) if preview_path.exists() else {},
        "frames": frames,
    }
    if out_path is not None:
        ensure_dir(out_path.parent)
        write_json(out_path, report)
    return report


def read_preview_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preview_row_by_tick(path: Path, tick: int) -> dict[str, str] | None:
    for row in read_preview_rows(path):
        try:
            if int(row.get("tick", -1)) == tick:
                return row
        except ValueError:
            continue
    return None


def read_events(scene_dir: Path) -> list[dict[str, object]]:
    path = scene_dir / "events.jsonl"
    if not path.exists():
        return []
    import json

    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _read_scene(scene_dir: Path) -> dict[str, object]:
    path = scene_dir / "scene.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing scene.json in {scene_dir}")
    return read_json(path)


def _int_value(value: object, default: int) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _float_value(value: object, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _is_numeric(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return value not in (None, "")
