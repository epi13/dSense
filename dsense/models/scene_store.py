from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dsense.manifest import project_path
from dsense.models.features import feature_manifest, read_numeric_preview_rows, summarize_rows
from dsense.perf import resolve_worker_count
from dsense.utils.files import ensure_dir, read_json, write_json
from dsense.utils.timebase import utc_now_iso


FEATURE_EXTRACTION_VERSION = "dsense-scene-feature-store-v1"
CachePolicy = Literal["auto", "rebuild", "off"]


@dataclass(frozen=True)
class SceneFeatureRecord:
    scene_id: str
    label: str
    accepted: bool
    created_utc: str
    scene_dir: str
    preview_rows: list[dict[str, float]]
    summary_features: dict[str, float]
    timeseries_features: dict[str, float]
    contrastive_features: dict[str, float]
    source_columns: list[str]

    @property
    def preview_row_count(self) -> int:
        return len(self.preview_rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "label": self.label,
            "accepted": self.accepted,
            "created_utc": self.created_utc,
            "scene_dir": self.scene_dir,
            "preview_rows": self.preview_rows,
            "summary_features": self.summary_features,
            "timeseries_features": self.timeseries_features,
            "contrastive_features": self.contrastive_features,
            "source_columns": self.source_columns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SceneFeatureRecord":
        return cls(
            scene_id=str(data.get("scene_id", "")),
            label=str(data.get("label", "unknown")),
            accepted=bool(data.get("accepted", True)),
            created_utc=str(data.get("created_utc", "")),
            scene_dir=str(data.get("scene_dir", "")),
            preview_rows=[{str(k): float(v) for k, v in dict(row).items()} for row in list(data.get("preview_rows", []))],
            summary_features={str(k): float(v) for k, v in dict(data.get("summary_features", {})).items()},
            timeseries_features={str(k): float(v) for k, v in dict(data.get("timeseries_features", {})).items()},
            contrastive_features={str(k): float(v) for k, v in dict(data.get("contrastive_features", {})).items()},
            source_columns=[str(column) for column in list(data.get("source_columns", []))],
        )


@dataclass
class SceneFeatureStore:
    project_name: str
    fingerprint: dict[str, object]
    scenes: list[SceneFeatureRecord]
    worker_count: int = 1
    cache_hit: bool = False
    created_utc: str = field(default_factory=utc_now_iso)

    @property
    def accepted_scenes(self) -> list[SceneFeatureRecord]:
        return [scene for scene in self.scenes if scene.accepted and scene.preview_rows]

    @property
    def scene_count(self) -> int:
        return len(self.accepted_scenes)

    @property
    def preview_row_count(self) -> int:
        return sum(scene.preview_row_count for scene in self.accepted_scenes)

    @property
    def source_columns(self) -> list[str]:
        return sorted({column for scene in self.accepted_scenes for column in scene.source_columns})

    def to_dict(self) -> dict[str, object]:
        return {
            "format": FEATURE_EXTRACTION_VERSION,
            "project_name": self.project_name,
            "created_utc": self.created_utc,
            "fingerprint": self.fingerprint,
            "worker_count": self.worker_count,
            "scene_count": self.scene_count,
            "preview_row_count": self.preview_row_count,
            "source_columns": self.source_columns,
            "scenes": [scene.to_dict() for scene in self.scenes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object], *, cache_hit: bool = False) -> "SceneFeatureStore":
        return cls(
            project_name=str(data.get("project_name", "")),
            fingerprint=dict(data.get("fingerprint", {})),
            scenes=[SceneFeatureRecord.from_dict(dict(scene)) for scene in list(data.get("scenes", []))],
            worker_count=int(data.get("worker_count", 1) or 1),
            cache_hit=cache_hit,
            created_utc=str(data.get("created_utc", "")) or utc_now_iso(),
        )


def feature_store_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "feature_store.json"


def dataset_fingerprint(
    project_name: str,
    *,
    channel_groups: list[str] | tuple[str, ...] | None = None,
    model_options: dict[str, object] | None = None,
) -> dict[str, object]:
    root = project_path(project_name)
    scenes = []
    accepted_count = 0
    preview_row_count_hint = 0
    for scene_path in sorted((root / "scenes").glob("scene_*/scene.json")):
        try:
            scene = read_json(scene_path)
        except (OSError, ValueError):
            continue
        preview_path = scene_path.parent / "preview.csv"
        scene_stat = _safe_stat(scene_path)
        preview_stat = _safe_stat(preview_path)
        accepted = scene.get("accepted") is not False
        accepted_count += 1 if accepted else 0
        preview_row_count_hint += _count_csv_data_rows(preview_path)
        scenes.append({
            "scene_id": str(scene.get("scene_id", scene_path.parent.name)),
            "label": str(scene.get("label", "unknown")),
            "accepted": accepted,
            "scene_json": scene_stat,
            "scene_digest": _json_digest({
                "scene_id": scene.get("scene_id", scene_path.parent.name),
                "label": scene.get("label", "unknown"),
                "accepted": accepted,
            }),
            "preview_csv": preview_stat,
            "preview_digest": _file_digest(preview_path) if preview_path.exists() else "",
        })
    payload = {
        "format": "dsense-dataset-fingerprint-v1",
        "project_name": project_name,
        "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
        "channel_groups": sorted(str(group) for group in (channel_groups or [])),
        "model_options": _canonical_options(model_options or {}),
        "scene_count": len(scenes),
        "accepted_scene_count": accepted_count,
        "preview_row_count": preview_row_count_hint,
        "scenes": scenes,
    }
    payload["hash"] = _json_digest(payload)
    return payload


def build_or_load_feature_store(
    project_name: str,
    *,
    workers: int | None = None,
    force: bool = False,
    cache_policy: CachePolicy = "auto",
    channel_groups: list[str] | tuple[str, ...] | None = None,
    model_options: dict[str, object] | None = None,
) -> SceneFeatureStore:
    resolved_workers = resolve_worker_count(workers)
    fingerprint = dataset_fingerprint(project_name, channel_groups=channel_groups, model_options=model_options)
    path = feature_store_path(project_name)
    if cache_policy not in {"auto", "rebuild", "off"}:
        raise ValueError(f"Unknown startup cache policy: {cache_policy}")
    if not force and cache_policy == "auto" and path.exists():
        try:
            cached = SceneFeatureStore.from_dict(read_json(path), cache_hit=True)
        except (OSError, ValueError):
            cached = None
        if cached is not None and same_fingerprint(cached.fingerprint, fingerprint):
            cached.worker_count = resolved_workers
            return cached
    store = build_feature_store(project_name, fingerprint=fingerprint, workers=resolved_workers)
    if cache_policy != "off":
        ensure_dir(path.parent)
        write_json(path, store.to_dict())
    return store


def build_feature_store(project_name: str, *, fingerprint: dict[str, object] | None = None, workers: int | None = None) -> SceneFeatureStore:
    root = project_path(project_name)
    resolved_workers = resolve_worker_count(workers)
    scene_paths = sorted((root / "scenes").glob("scene_*/scene.json"))
    tasks = [(str(path), str(root)) for path in scene_paths]
    if resolved_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=resolved_workers) as pool:
            records = list(pool.map(_extract_scene_feature, tasks))
    else:
        records = [_extract_scene_feature(task) for task in tasks]
    scenes = [record for record in records if record is not None]
    return SceneFeatureStore(
        project_name=project_name,
        fingerprint=fingerprint or dataset_fingerprint(project_name),
        scenes=scenes,
        worker_count=resolved_workers,
        cache_hit=False,
    )


def feature_manifest_from_store(store: SceneFeatureStore, feature_name: str = "summary_features") -> dict[str, object]:
    features = [dict(getattr(scene, feature_name)) for scene in store.accepted_scenes if getattr(scene, feature_name)]
    rows = [row for scene in store.accepted_scenes for row in scene.preview_rows]
    manifest = feature_manifest(features, rows)
    manifest["dataset_fingerprint"] = store.fingerprint
    manifest["feature_store"] = {
        "format": FEATURE_EXTRACTION_VERSION,
        "cache_hit": store.cache_hit,
        "worker_count": store.worker_count,
        "scene_count": store.scene_count,
        "preview_row_count": store.preview_row_count,
    }
    return manifest


def same_fingerprint(left: dict[str, object] | None, right: dict[str, object] | None) -> bool:
    return bool(left and right and str(left.get("hash", "")) and left.get("hash") == right.get("hash"))


def artifact_matches_fingerprint(path: Path, fingerprint: dict[str, object], *, model_version: str | None = None) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return False
    manifest = dict(data.get("feature_manifest", {}))
    if not same_fingerprint(dict(manifest.get("dataset_fingerprint", {})), fingerprint):
        return False
    if model_version is None:
        return True
    return str(manifest.get("model_version", "")) == model_version


def evaluation_matches_fingerprint(path: Path, fingerprint: dict[str, object], mode: str) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return False
    return same_fingerprint(dict(data.get("dataset_fingerprint", {})), fingerprint) and str(data.get("evaluation_mode", "full")) == mode


def _extract_scene_feature(task: tuple[str, str]) -> SceneFeatureRecord | None:
    scene_path = Path(task[0])
    root = Path(task[1])
    try:
        scene = read_json(scene_path)
    except (OSError, ValueError):
        return None
    if scene.get("accepted") is False:
        return SceneFeatureRecord(
            scene_id=str(scene.get("scene_id", scene_path.parent.name)),
            label=str(scene.get("label", "unknown")),
            accepted=False,
            created_utc=str(scene.get("created_utc", "")),
            scene_dir=str(scene_path.parent.relative_to(root)),
            preview_rows=[],
            summary_features={},
            timeseries_features={},
            contrastive_features={},
            source_columns=[],
        )
    preview_path = scene_path.parent / "preview.csv"
    if not preview_path.exists():
        return None
    rows = read_numeric_preview_rows(preview_path)
    if not rows:
        return None
    from dsense.contrastive import extract_contrastive_features
    from dsense.timeseries import extract_timeseries_features

    return SceneFeatureRecord(
        scene_id=str(scene.get("scene_id", scene_path.parent.name)),
        label=str(scene.get("label", "unknown")),
        accepted=True,
        created_utc=str(scene.get("created_utc", "")),
        scene_dir=str(scene_path.parent.relative_to(root)),
        preview_rows=rows,
        summary_features=summarize_rows(rows),
        timeseries_features=extract_timeseries_features(rows),
        contrastive_features=extract_contrastive_features(rows),
        source_columns=sorted({column for row in rows for column in row}),
    )


def _safe_stat(path: Path) -> dict[str, int]:
    try:
        stat = path.stat()
    except OSError:
        return {"size": 0, "mtime_ns": 0}
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        return ""
    return hasher.hexdigest()


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_options(options: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(options, sort_keys=True, default=str))


def _count_csv_data_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0
