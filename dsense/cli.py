from __future__ import annotations

import argparse, csv
from pathlib import Path
from .manifest import init_project, scan_channels, project_path, load_manifest, allocate_scene_id
from .recorder import record_scene
from .wizard import guided_scene
from .utils.files import read_json
from .autotest import validate_dataset, print_validation_report


def cmd_init(args):
    root = init_project(args.project_name)
    print(root / "manifest.json"); print(root / "channels.json"); print(root / "scenes"); print(root / "exports")


def cmd_scan(args):
    for ch in scan_channels():
        status = "available" if ch["available"] else f"unavailable ({ch['reason']})"
        print(f"{ch['id']}: {status} - {ch['name']}")


def cmd_record_baseline(args):
    init_project(args.project_name)
    scene_id = allocate_scene_id(args.project_name)
    scene = record_scene(project_path(args.project_name) / "scenes" / scene_id, scene_id, "baseline_idle", args.duration, args.tick_hz, 0, args.duration, 0, args.notes)
    print(f"Recorded {scene_id}: confidence={scene['quality']['confidence']}")


def cmd_scene(args):
    init_project(args.project_name)
    duration = args.duration or (args.pre_roll + args.action + args.post_roll)
    guided_scene(args.project_name, args.label, duration, args.pre_roll, args.action, args.post_roll, args.repeat, args.notes, args.tick_hz, args.yes)


def cmd_list(args):
    root = project_path(args.project_name) / "scenes"
    if not root.exists():
        raise SystemExit("No scenes found. Run dsense init first.")
    for p in sorted(root.glob("scene_*/scene.json")):
        s = read_json(p)
        print(f"{s['scene_id']}  {s['label']}  {s['duration_ms']}ms  accepted={s['accepted']}  confidence={s['quality']['confidence']}")


def cmd_export(args):
    root = project_path(args.project_name)
    out = root / "exports" / "preview_index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "label", "duration_ms", "tick_hz", "confidence", "preview_csv"])
        writer.writeheader()
        for p in sorted((root / "scenes").glob("scene_*/scene.json")):
            s = read_json(p)
            writer.writerow({"scene_id": s["scene_id"], "label": s["label"], "duration_ms": s["duration_ms"], "tick_hz": s["tick_hz"], "confidence": s["quality"]["confidence"], "preview_csv": str(p.parent / "preview.csv")})
    print(f"Wrote {out}")


def cmd_validate(args):
    """Validate all scenes in a project and print a detailed report."""
    result = validate_dataset(args.project_name)
    print_validation_report(result, verbose=args.verbose)


def build_parser():
    p = argparse.ArgumentParser(prog="dsense", description="dSense Scene Wizard")
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("init"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_init)
    sp = sub.add_parser("scan"); sp.set_defaults(func=cmd_scan)
    sp = sub.add_parser("record-baseline"); sp.add_argument("project_name"); sp.add_argument("--duration", type=float, default=30); sp.add_argument("--tick-hz", type=int, default=100); sp.add_argument("--notes", default=""); sp.set_defaults(func=cmd_record_baseline)
    sp = sub.add_parser("scene"); sp.add_argument("project_name"); sp.add_argument("--label", required=True); sp.add_argument("--duration", type=float); sp.add_argument("--pre-roll", type=float, default=2); sp.add_argument("--action", type=float, default=5); sp.add_argument("--post-roll", type=float, default=3); sp.add_argument("--repeat", type=int, default=1); sp.add_argument("--notes", default=""); sp.add_argument("--tick-hz", type=int, default=100); sp.add_argument("--yes", action="store_true", help="accept captures without prompt"); sp.set_defaults(func=cmd_scene)
    sp = sub.add_parser("list-scenes"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_list)
    sp = sub.add_parser("export-preview"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_export)
    sp = sub.add_parser("validate"); sp.add_argument("project_name"); sp.add_argument("--verbose", "-v", action="store_true", help="show detailed error messages"); sp.set_defaults(func=cmd_validate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
