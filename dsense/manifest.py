from __future__ import annotations

from pathlib import Path
from .channels import default_channels
from .utils.files import ensure_dir, write_json, read_json
from .utils.timebase import utc_now_iso

DATASETS = Path("datasets")
DEFAULT_PROJECT = "base"


def project_path(name: str) -> Path:
    return DATASETS / name


def init_project(name: str) -> Path:
    root = project_path(name)
    ensure_dir(root / "scenes")
    ensure_dir(root / "exports")
    manifest = root / "manifest.json"
    if not manifest.exists():
        write_json(manifest, {"project_name": name, "created_utc": utc_now_iso(), "format": "dsense-scene-v0", "next_scene": 1})
    write_json(root / "channels.json", {"channels": scan_channels()})
    return root


def load_manifest(name: str) -> dict[str, object]:
    path = project_path(name) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Project not found: {project_path(name)}. Run 'dsense init {name}' first.")
    return read_json(path)


def save_manifest(name: str, data: dict[str, object]) -> None:
    write_json(project_path(name) / "manifest.json", data)


def allocate_scene_id(name: str) -> str:
    data = load_manifest(name)
    n = int(data.get("next_scene", 1))
    scenes_root = project_path(name) / "scenes"
    while (scenes_root / f"scene_{n:06d}").exists():
        n += 1
    data["next_scene"] = n + 1
    save_manifest(name, data)
    return f"scene_{n:06d}"


def scan_channels() -> list[dict[str, object]]:
    out = []
    for ch in default_channels():
        try:
            available = ch.available()
            reason = "ok" if available else "unavailable"
        except Exception as exc:
            available, reason = False, str(exc)
        out.append({"id": ch.id, "name": ch.name, "rate_hz": ch.rate_hz, "bit": ch.bit, "available": available, "reason": reason})
    return out
