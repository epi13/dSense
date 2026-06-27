from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
from .baseline import train_and_save_project_baseline
from .classifier import load_project_classifier, predict_scene, train_and_save_project_classifier
from .gemma_edge import gemma_edge_status
from .frame import FRAME_SIZE, frame_to_dict
from .manifest import DEFAULT_PROJECT, init_project, scan_channels, project_path, load_manifest, allocate_scene_id
from .models.evaluation import evaluate_project_scenes, evaluation_report_path, print_evaluation_report
from .recorder import record_scene
from .tui import CaptureConfig, run_tui
from .wizard import guided_scene
from .utils.files import read_json
from .autotest import validate_dataset, print_validation_report
from .transfer import compare_transfer_bundle, export_transfer_bundle, transfer_bundle_path


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
    if args.tui:
        run_tui(CaptureConfig(
            project_name=args.project_name,
            label=args.label,
            duration=duration,
            pre_roll=args.pre_roll,
            action=args.action,
            post_roll=args.post_roll,
            repeat=args.repeat,
            tick_hz=args.tick_hz,
            notes=args.notes,
        ))
        return
    guided_scene(args.project_name, args.label, duration, args.pre_roll, args.action, args.post_roll, args.repeat, args.notes, args.tick_hz, args.yes)


def cmd_tui(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    duration = args.duration or (args.pre_roll + args.action + args.post_roll)
    run_tui(CaptureConfig(
        project_name=project_name,
        label=args.label,
        duration=duration,
        pre_roll=args.pre_roll,
        action=args.action,
        post_roll=args.post_roll,
        repeat=args.repeat,
        tick_hz=args.tick_hz,
        notes=args.notes,
    ))


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


def cmd_train_classifier(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    _require_valid_dataset(project_name, args.require_valid)
    model = train_and_save_project_classifier(project_name)
    print(f"Trained classifier for {project_name}: {model.scene_count} scenes, {model.baseline_scene_count} baseline scenes")
    print(project_path(project_name) / "exports" / "classifier.json")


def cmd_train_baseline(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    _require_valid_dataset(project_name, args.require_valid)
    model = train_and_save_project_baseline(project_name)
    print(f"Trained baseline for {project_name}: {model.scene_count} baseline scenes, {len(model.channels)} channels")
    print(project_path(project_name) / "exports" / "baseline_model.json")


def cmd_export_transfer(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    _require_valid_dataset(project_name, args.require_valid)
    bundle = export_transfer_bundle(project_name)
    print(f"Exported transfer bundle for {project_name}: {bundle['total_scenes']} scenes")
    print(transfer_bundle_path(project_name))


def cmd_evaluate_scenes(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    report = evaluate_project_scenes(project_name)
    print_evaluation_report(report)
    print(evaluation_report_path(project_name))


def cmd_classify_scene(args):
    scene_dir = Path(args.scene_dir)
    preview_path = scene_dir / "preview.csv"
    if not preview_path.exists():
        raise SystemExit(f"Missing preview.csv in {scene_dir}")
    model = load_project_classifier(args.project)
    result = predict_scene(model, preview_path)
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_inspect_frame(args):
    scene_dir = Path(args.scene_dir)
    frames_path = scene_dir / "frames.ds64"
    if not frames_path.exists():
        raise SystemExit(f"Missing frames.ds64 in {scene_dir}")
    offset = args.tick * FRAME_SIZE
    with frames_path.open("rb") as handle:
        handle.seek(offset)
        frame = handle.read(FRAME_SIZE)
    if len(frame) != FRAME_SIZE:
        raise SystemExit(f"Tick {args.tick} is outside {frames_path}")
    result: dict[str, object] = {"frame": frame_to_dict(frame)}
    preview_row = _preview_row(scene_dir / "preview.csv", args.tick)
    if preview_row is not None:
        result["preview"] = preview_row
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_replay(args):
    scene_dir = Path(args.scene_dir)
    scene_path = scene_dir / "scene.json"
    preview_path = scene_dir / "preview.csv"
    events_path = scene_dir / "events.jsonl"
    if not scene_path.exists() or not preview_path.exists():
        raise SystemExit(f"Scene must contain scene.json and preview.csv: {scene_dir}")
    scene = read_json(scene_path)
    rows = _preview_rows(preview_path)
    events = _event_rows(events_path)
    print(f"Replay {scene.get('scene_id', scene_dir.name)} label={scene.get('label', 'unknown')} rows={len(rows)} events={len(events)}")
    if events:
        print("Events:")
        for event in events:
            print(f"  {event.get('t_ms', '?')}ms {event.get('event', 'unknown')}")
    if rows:
        print("Preview:")
        for row in rows[: args.limit]:
            print(f"  tick={row.get('tick')} t_ns={row.get('t_ns')} dt_ns={row.get('dt_ns')} sleep_drift_ns={row.get('sleep_drift_ns')} process_ns_estimate={row.get('process_ns_estimate')}")
        if len(rows) > args.limit:
            print(f"  ... {len(rows) - args.limit} more rows")


def cmd_compare_transfer(args):
    if args.bundle is None:
        project_name = DEFAULT_PROJECT
        bundle = args.project_or_bundle
    else:
        project_name = args.project_or_bundle or DEFAULT_PROJECT
        bundle = args.bundle
    init_project(project_name)
    result = compare_transfer_bundle(project_name, Path(bundle))
    print(f"Compatibility: {result['compatibility']}")
    print(f"Transfer risk: {result['transfer_risk']}")
    print(f"Missing channels: {', '.join(result['missing_channels']) or 'none'}")
    print(f"Extra channels: {', '.join(result['extra_channels']) or 'none'}")
    print(f"Max baseline drift: {result['baseline_drift']['max_drift']:.3f}")


def cmd_gemma_status(args):
    status = gemma_edge_status()
    state = "enabled" if status["enabled"] else "disabled"
    print(f"Gemma Edge: {state}")
    print(f"Model: {status['model']}")
    print(f"Command: {status['command'] or 'not set'}")
    print(f"Mode: {status['mode']}")


def _require_valid_dataset(project_name: str, require_valid: bool) -> None:
    if not require_valid:
        return
    result = validate_dataset(project_name)
    if result.error_count:
        print_validation_report(result, verbose=True)
        raise SystemExit(f"Validation failed for {project_name}: {result.error_count} errors")


def _preview_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _preview_row(path: Path, tick: int) -> dict[str, str] | None:
    for row in _preview_rows(path):
        try:
            if int(row.get("tick", -1)) == tick:
                return row
        except ValueError:
            continue
    return None


def _event_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def build_parser():
    p = argparse.ArgumentParser(prog="dsense", description="dSense Scene Wizard")
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("init"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_init)
    sp = sub.add_parser("scan"); sp.set_defaults(func=cmd_scan)
    sp = sub.add_parser("record-baseline"); sp.add_argument("project_name"); sp.add_argument("--duration", type=float, default=30); sp.add_argument("--tick-hz", type=int, default=100); sp.add_argument("--notes", default=""); sp.set_defaults(func=cmd_record_baseline)
    sp = sub.add_parser("scene"); sp.add_argument("project_name"); sp.add_argument("--label", required=True); sp.add_argument("--duration", type=float); sp.add_argument("--pre-roll", type=float, default=2); sp.add_argument("--action", type=float, default=5); sp.add_argument("--post-roll", type=float, default=3); sp.add_argument("--repeat", type=int, default=1); sp.add_argument("--notes", default=""); sp.add_argument("--tick-hz", type=int, default=100); sp.add_argument("--yes", action="store_true", help="accept captures without prompt"); sp.add_argument("--tui", action="store_true", help="record with the full-screen interaction recorder"); sp.set_defaults(func=cmd_scene)
    sp = sub.add_parser("tui"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to open (default: {DEFAULT_PROJECT})"); sp.add_argument("--label", default="user_interaction"); sp.add_argument("--duration", type=float); sp.add_argument("--pre-roll", type=float, default=2); sp.add_argument("--action", type=float, default=5); sp.add_argument("--post-roll", type=float, default=3); sp.add_argument("--repeat", type=int, default=1); sp.add_argument("--notes", default=""); sp.add_argument("--tick-hz", type=int, default=100); sp.set_defaults(func=cmd_tui)
    sp = sub.add_parser("list-scenes"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_list)
    sp = sub.add_parser("export-preview"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_export)
    sp = sub.add_parser("validate"); sp.add_argument("project_name"); sp.add_argument("--verbose", "-v", action="store_true", help="show detailed error messages"); sp.set_defaults(func=cmd_validate)
    sp = sub.add_parser("train-baseline"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to train (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_train_baseline)
    sp.add_argument("--require-valid", action="store_true", help="fail before training if dataset validation has errors")
    sp = sub.add_parser("train-classifier"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to train (default: {DEFAULT_PROJECT})"); sp.add_argument("--require-valid", action="store_true", help="fail before training if dataset validation has errors"); sp.set_defaults(func=cmd_train_classifier)
    sp = sub.add_parser("export-transfer"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to export (default: {DEFAULT_PROJECT})"); sp.add_argument("--require-valid", action="store_true", help="fail before export if dataset validation has errors"); sp.set_defaults(func=cmd_export_transfer)
    sp = sub.add_parser("evaluate-scenes"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to evaluate (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_evaluate_scenes)
    sp = sub.add_parser("classify-scene"); sp.add_argument("scene_dir"); sp.add_argument("--project", default=DEFAULT_PROJECT, help=f"classifier project to use (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_classify_scene)
    sp = sub.add_parser("inspect-frame"); sp.add_argument("scene_dir"); sp.add_argument("--tick", type=int, required=True); sp.set_defaults(func=cmd_inspect_frame)
    sp = sub.add_parser("replay"); sp.add_argument("scene_dir"); sp.add_argument("--limit", type=int, default=10, help="preview rows to print"); sp.set_defaults(func=cmd_replay)
    sp = sub.add_parser("compare-transfer"); sp.add_argument("project_or_bundle", help="project name or transfer bundle JSON path"); sp.add_argument("bundle", nargs="?", help="transfer bundle JSON path"); sp.set_defaults(func=cmd_compare_transfer)
    sp = sub.add_parser("gemma-status"); sp.set_defaults(func=cmd_gemma_status)
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = ["tui"]
    args = build_parser().parse_args(argv)
    args.func(args)
