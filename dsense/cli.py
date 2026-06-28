from __future__ import annotations

import argparse, csv, json, sys
import curses
from pathlib import Path
from .baseline import train_and_save_project_baseline
from .channels import parse_channel_groups
from .classifier import train_and_save_project_classifier
from .doctor import doctor_ok, print_doctor_report, run_doctor
from .gemma_edge import gemma_edge_status
from .manifest import DEFAULT_PROJECT, init_project, scan_channels, project_path, allocate_scene_id
from .models.evaluation import evaluate_project_scenes, evaluation_report_path, print_evaluation_report
from .models.feature_export import extract_project_features, features_path, rank_project_channels
from .orbiters import evaluate_project_orbiters, run_scene_orbiters
from .privacy import build_privacy_report, print_privacy_report, privacy_report_path
from .replay import (
    classify_existing_scene,
    export_scene_json,
    inspect_frame,
    inspect_scene,
    replay_scene,
    resolve_scene_dir,
)
from .recorder import record_scene
from .tui import CaptureConfig, run_tui
from .trace import export_trace, trace_path, viewer_path, write_scene_viewer
from .wizard import guided_scene
from .utils.files import read_json
from .autotest import validate_dataset, print_validation_report
from .transfer import compare_transfer_bundle, export_transfer_bundle, transfer_bundle_path
from .watcher import label_candidate, run_rolling_watcher, run_watcher_scan


def cmd_init(args):
    root = init_project(args.project_name)
    print(root / "manifest.json"); print(root / "channels.json"); print(root / "scenes"); print(root / "exports")


def cmd_scan(args):
    for ch in scan_channels(advanced=args.advanced):
        status = "available" if ch["available"] else f"unavailable ({ch['reason']})"
        print(f"{ch['id']} [{ch.get('group', 'portable')}]: {status} - {ch['name']}")


def cmd_record_baseline(args):
    init_project(args.project_name)
    scene_id = allocate_scene_id(args.project_name)
    groups = parse_channel_groups(args.channels)
    scene = record_scene(project_path(args.project_name) / "scenes" / scene_id, scene_id, "baseline_idle", args.duration, args.tick_hz, 0, args.duration, 0, args.notes, channel_groups=groups)
    print(f"Recorded {scene_id}: confidence={scene['quality']['confidence']}")


def cmd_scene(args):
    init_project(args.project_name)
    duration = args.duration or (args.pre_roll + args.action + args.post_roll)
    groups = parse_channel_groups(args.channels)
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
    guided_scene(args.project_name, args.label, duration, args.pre_roll, args.action, args.post_roll, args.repeat, args.notes, args.tick_hz, args.yes, channel_groups=groups)


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


def cmd_doctor(args):
    checks = run_doctor(args.project_name)
    print_doctor_report(checks)
    if not doctor_ok(checks):
        raise SystemExit(1)


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
    if args.redact:
        report = build_privacy_report(project_name)
        print("Sharing summary before export:")
        print(f"  scenes={report['scene_count']} labels={report['label_count']} warnings={len(report.get('warnings', []))}")
        print("  redaction removes project name, timestamps, label profiles, label counts, notes, and raw scenes")
    bundle = export_transfer_bundle(project_name, redact=args.redact)
    mode = "redacted transfer bundle" if args.redact else "transfer bundle"
    print(f"Exported {mode} for {project_name}: {bundle['total_scenes']} scenes")
    print(transfer_bundle_path(project_name))


def cmd_privacy_report(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    out_path = Path(args.out) if args.out else None
    report = build_privacy_report(project_name, out_path=out_path)
    print_privacy_report(report)
    print(out_path or privacy_report_path(project_name))


def cmd_evaluate_scenes(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    out_path = Path(args.out) if args.out else None
    report = evaluate_project_scenes(project_name, out_path=out_path)
    print_evaluation_report(report)
    print(out_path or evaluation_report_path(project_name))


def cmd_extract_features(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    out_path = Path(args.out) if args.out else None
    report = extract_project_features(project_name, out_path=out_path)
    manifest = dict(report.get("feature_manifest", {}))
    print(f"Extracted features for {project_name}: {report['scene_count']} scenes, {manifest.get('feature_count', 0)} features")
    print(out_path or features_path(project_name))


def cmd_rank_channels(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    ranking = rank_project_channels(project_name)
    print(f"Channel ranking: {project_name}")
    if not ranking:
        print("  none")
        return
    print(f"{'channel':<28} {'score':<12} best feature")
    for item in ranking[: args.limit]:
        print(f"{str(item.get('channel', 'unknown')):<28} {float(item.get('score', 0.0)):<12.6f} {item.get('best_feature', '')}")


def cmd_classify_scene(args):
    project_name, scene_dir = _resolve_scene_args(args.target, args.scene_id, args.project)
    result = classify_existing_scene(project_name, scene_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_inspect_frame(args):
    _, scene_dir = _resolve_scene_args(args.target, args.scene_id, args.project)
    result = inspect_frame(scene_dir, args.tick)
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_inspect_scene(args):
    _, scene_dir = _resolve_scene_args(args.target, args.scene_id, args.project)
    summary = inspect_scene(scene_dir)
    print(f"{summary['scene_id']}  {summary['label']}  accepted={summary['accepted']}  frames={summary['frame_count']}  events={summary['event_count']}")
    print(f"duration_ms={summary['duration_ms']} tick_hz={summary['tick_hz']} preview_rows={summary['preview_rows']}")
    quality = summary.get("quality", {})
    if isinstance(quality, dict):
        print(f"confidence={quality.get('confidence', '?')} checksum_ok={quality.get('checksum_ok', '?')} frame_size_valid={quality.get('frame_size_valid', '?')}")
    if summary.get("notes"):
        print(f"notes={summary['notes']}")


def cmd_replay_scene(args):
    project_name, scene_dir = _resolve_scene_args(args.target, args.scene_id, args.project)
    result = replay_scene(project_name, scene_dir, limit=args.limit)
    scene = dict(result["scene"])
    prediction = dict(result["classifier_prediction"])
    print(f"Replay {scene.get('scene_id')} label={scene.get('label')} rows={scene.get('preview_rows')} recorded_events={len(result['recorded_events'])} detector_events={len(result['detector_events'])}")
    print(f"classifier={prediction.get('label')} confidence={prediction.get('confidence')} distance={prediction.get('distance')}")
    state = dict(result["detector_state"])
    print(f"detector={state.get('status')} channel={state.get('channel')} score={state.get('score')}")
    if result["detector_events"]:
        print("Detector events:")
        for event in list(result["detector_events"])[: args.limit]:
            print(f"  {event.get('t_ms', '?')}ms {event.get('event', 'unknown')} {event.get('channel', '')} score={event.get('score', '')}")


def cmd_export_scene_json(args):
    project_name, scene_dir = _resolve_scene_args(args.target, args.scene_id, args.project)
    out_path = Path(args.out) if args.out else project_path(project_name) / "exports" / f"{scene_dir.name}.json"
    report = export_scene_json(project_name, scene_dir, out_path)
    print(f"Exported {report['summary']['scene_id']}: {len(report['frames'])} frames")
    print(out_path)


def cmd_export_trace(args):
    project_name, scene_dir = _resolve_scene_args(args.target, args.scene_id, args.project)
    out_path = Path(args.out) if args.out else trace_path(scene_dir)
    trace = export_trace(project_name, scene_dir, out_path)
    print(f"Exported trace for {trace['scene']['scene_id']}: {len(trace['tracks'])} tracks, {len(trace['events'])} events")
    print(out_path)


def cmd_view_scene(args):
    project_name, scene_dir = _resolve_scene_args(args.target, args.scene_id, args.project)
    out_path = Path(args.out) if args.out else viewer_path(scene_dir)
    path = write_scene_viewer(project_name, scene_dir, out_path=out_path, open_browser=not args.no_open)
    print(path)


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


def cmd_watcher(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    if args.rolling:
        result = run_rolling_watcher(
            project_name,
            pre_seconds=args.pre,
            post_seconds=args.post,
            tick_hz=args.tick_hz,
            cooldown_seconds=args.cooldown,
            duration=args.duration,
            prompt_label=args.prompt_label,
        )
        print(f"Rolling watcher saved {len(result['saved'])} anomaly windows")
        print(result["session_path"])
        for saved in result["saved"]:
            scene = dict(saved["scene"])
            event = dict(saved["event"])
            print(f"{scene.get('scene_id')} score={event.get('anomaly_score')} label={scene.get('label')} accepted={scene.get('accepted')}")
        return
    result = run_watcher_scan(project_name, duration=args.duration or 5.0, tick_hz=args.tick_hz)
    scene = dict(result["scene"])
    print(f"Watcher scan saved {scene.get('scene_id')} label={scene.get('label')} accepted={scene.get('accepted')}")
    print(result["watcher_events_path"])


def cmd_label_candidate(args):
    scene = label_candidate(args.project_name, args.scene_id, args.label, notes=args.notes)
    print(f"Labeled {scene['scene_id']}: {scene.get('previous_label')} -> {scene['label']}")


def cmd_orbiter_run(args):
    summary = run_scene_orbiters(args.project_name, args.scene_id)
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_orbiter_evaluate(args):
    result = evaluate_project_orbiters(args.project_name)
    print(f"Orbiter evaluation: {args.project_name}")
    print(f"Evaluated: {result['evaluated']}  Matches: {result['matches']}  Accuracy: {result['accuracy']}")
    for summary in result["summaries"][: args.limit]:
        print(f"{summary['scene_id']} actual={summary['actual_label']} predicted={summary['predicted_label']} match={summary['matches_actual_label']} confidence={summary['confidence']}")


def _require_valid_dataset(project_name: str, require_valid: bool) -> None:
    if not require_valid:
        return
    result = validate_dataset(project_name)
    if result.error_count:
        print_validation_report(result, verbose=True)
        raise SystemExit(f"Validation failed for {project_name}: {result.error_count} errors")


def _resolve_scene_args(target: str, scene_id: str | None, project: str | None = None) -> tuple[str, Path]:
    if scene_id is not None:
        project_name = target
        scene_dir = resolve_scene_dir(project_name, scene_id)
    else:
        project_name = project or DEFAULT_PROJECT
        scene_dir = resolve_scene_dir(target)
    if not scene_dir.exists():
        raise SystemExit(f"Scene not found: {scene_dir}")
    return project_name, scene_dir


def build_parser():
    p = argparse.ArgumentParser(prog="dsense", description="dSense Scene Wizard")
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("init"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_init)
    sp = sub.add_parser("doctor"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to check (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_doctor)
    sp = sub.add_parser("scan"); sp.add_argument("--advanced", action="store_true", help="include linux and experimental adapters"); sp.set_defaults(func=cmd_scan)
    sp = sub.add_parser("record-baseline"); sp.add_argument("project_name"); sp.add_argument("--duration", type=float, default=30); sp.add_argument("--tick-hz", type=int, default=100); sp.add_argument("--notes", default=""); sp.add_argument("--channels", default="portable", help="channel groups, e.g. portable or portable,linux"); sp.set_defaults(func=cmd_record_baseline)
    sp = sub.add_parser("scene"); sp.add_argument("project_name"); sp.add_argument("--label", required=True); sp.add_argument("--duration", type=float); sp.add_argument("--pre-roll", type=float, default=2); sp.add_argument("--action", type=float, default=5); sp.add_argument("--post-roll", type=float, default=3); sp.add_argument("--repeat", type=int, default=1); sp.add_argument("--notes", default=""); sp.add_argument("--tick-hz", type=int, default=100); sp.add_argument("--channels", default="portable", help="channel groups, e.g. portable or portable,linux"); sp.add_argument("--yes", action="store_true", help="accept captures without prompt"); sp.add_argument("--tui", action="store_true", help="record with the full-screen interaction recorder"); sp.set_defaults(func=cmd_scene)
    sp = sub.add_parser("tui"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to open (default: {DEFAULT_PROJECT})"); sp.add_argument("--label", default="user_interaction"); sp.add_argument("--duration", type=float); sp.add_argument("--pre-roll", type=float, default=2); sp.add_argument("--action", type=float, default=5); sp.add_argument("--post-roll", type=float, default=3); sp.add_argument("--repeat", type=int, default=1); sp.add_argument("--notes", default=""); sp.add_argument("--tick-hz", type=int, default=100); sp.set_defaults(func=cmd_tui)
    sp = sub.add_parser("list-scenes"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_list)
    sp = sub.add_parser("export-preview"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_export)
    sp = sub.add_parser("validate"); sp.add_argument("project_name"); sp.add_argument("--verbose", "-v", action="store_true", help="show detailed error messages"); sp.set_defaults(func=cmd_validate)
    sp = sub.add_parser("train-baseline"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to train (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_train_baseline)
    sp.add_argument("--require-valid", action="store_true", help="fail before training if dataset validation has errors")
    sp = sub.add_parser("train-classifier"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to train (default: {DEFAULT_PROJECT})"); sp.add_argument("--require-valid", action="store_true", help="fail before training if dataset validation has errors"); sp.set_defaults(func=cmd_train_classifier)
    sp = sub.add_parser("export-transfer"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to export (default: {DEFAULT_PROJECT})"); sp.add_argument("--require-valid", action="store_true", help="fail before export if dataset validation has errors"); sp.add_argument("--redact", action="store_true", help="write a privacy-redacted safe transfer bundle"); sp.set_defaults(func=cmd_export_transfer)
    sp = sub.add_parser("privacy-report"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to inspect (default: {DEFAULT_PROJECT})"); sp.add_argument("--out", help="write privacy report JSON to this path"); sp.set_defaults(func=cmd_privacy_report)
    sp = sub.add_parser("evaluate-scenes"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to evaluate (default: {DEFAULT_PROJECT})"); sp.add_argument("--out", help="write report JSON to this path"); sp.set_defaults(func=cmd_evaluate_scenes)
    sp = sub.add_parser("extract-features"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to extract (default: {DEFAULT_PROJECT})"); sp.add_argument("--out", help="write feature JSON to this path"); sp.set_defaults(func=cmd_extract_features)
    sp = sub.add_parser("rank-channels"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to rank (default: {DEFAULT_PROJECT})"); sp.add_argument("--limit", type=int, default=10, help="number of channels to print"); sp.set_defaults(func=cmd_rank_channels)
    sp = sub.add_parser("inspect-scene"); sp.add_argument("target", help="project name or scene directory"); sp.add_argument("scene_id", nargs="?", help="scene id when target is a project"); sp.add_argument("--project", default=DEFAULT_PROJECT, help=f"classifier project for direct scene paths (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_inspect_scene)
    sp = sub.add_parser("inspect-frame"); sp.add_argument("target", help="project name or scene directory"); sp.add_argument("scene_id", nargs="?", help="scene id when target is a project"); sp.add_argument("--tick", type=int, required=True); sp.add_argument("--project", default=DEFAULT_PROJECT, help=f"project for direct scene paths (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_inspect_frame)
    sp = sub.add_parser("classify-scene"); sp.add_argument("target", help="project name or scene directory"); sp.add_argument("scene_id", nargs="?", help="scene id when target is a project"); sp.add_argument("--project", default=DEFAULT_PROJECT, help=f"classifier project for direct scene paths (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_classify_scene)
    sp = sub.add_parser("replay-scene"); sp.add_argument("target", help="project name or scene directory"); sp.add_argument("scene_id", nargs="?", help="scene id when target is a project"); sp.add_argument("--project", default=DEFAULT_PROJECT, help=f"project for direct scene paths (default: {DEFAULT_PROJECT})"); sp.add_argument("--limit", type=int, default=10, help="detector events to print"); sp.set_defaults(func=cmd_replay_scene)
    sp = sub.add_parser("replay"); sp.add_argument("target", help="project name or scene directory"); sp.add_argument("scene_id", nargs="?", help="scene id when target is a project"); sp.add_argument("--project", default=DEFAULT_PROJECT, help=f"project for direct scene paths (default: {DEFAULT_PROJECT})"); sp.add_argument("--limit", type=int, default=10, help="detector events to print"); sp.set_defaults(func=cmd_replay_scene)
    sp = sub.add_parser("export-scene-json"); sp.add_argument("target", help="project name or scene directory"); sp.add_argument("scene_id", nargs="?", help="scene id when target is a project"); sp.add_argument("--project", default=DEFAULT_PROJECT, help=f"project for direct scene paths (default: {DEFAULT_PROJECT})"); sp.add_argument("--out", help="write debug JSON to this path"); sp.set_defaults(func=cmd_export_scene_json)
    sp = sub.add_parser("export-trace"); sp.add_argument("target", help="project name or scene directory"); sp.add_argument("scene_id", nargs="?", help="scene id when target is a project"); sp.add_argument("--project", default=DEFAULT_PROJECT, help=f"project for direct scene paths (default: {DEFAULT_PROJECT})"); sp.add_argument("--out", help="write trace JSON to this path"); sp.set_defaults(func=cmd_export_trace)
    sp = sub.add_parser("view-scene"); sp.add_argument("target", help="project name or scene directory"); sp.add_argument("scene_id", nargs="?", help="scene id when target is a project"); sp.add_argument("--project", default=DEFAULT_PROJECT, help=f"project for direct scene paths (default: {DEFAULT_PROJECT})"); sp.add_argument("--out", help="write viewer HTML to this path"); sp.add_argument("--no-open", action="store_true", help="write the viewer without opening a browser"); sp.set_defaults(func=cmd_view_scene)
    sp = sub.add_parser("compare-transfer"); sp.add_argument("project_or_bundle", help="project name or transfer bundle JSON path"); sp.add_argument("bundle", nargs="?", help="transfer bundle JSON path"); sp.set_defaults(func=cmd_compare_transfer)
    sp = sub.add_parser("watcher"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to watch (default: {DEFAULT_PROJECT})"); sp.add_argument("--rolling", action="store_true", help="use rolling anomaly windows instead of scan recording"); sp.add_argument("--pre", type=float, default=5.0, help="seconds to keep before trigger"); sp.add_argument("--post", type=float, default=10.0, help="seconds to save after trigger"); sp.add_argument("--cooldown", type=float, default=30.0, help="seconds before saving another window"); sp.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 means indefinitely in rolling mode"); sp.add_argument("--tick-hz", type=int, default=50); sp.add_argument("--prompt-label", action="store_true", help="prompt for a label when an anomaly is saved"); sp.set_defaults(func=cmd_watcher)
    sp = sub.add_parser("label-candidate"); sp.add_argument("project_name"); sp.add_argument("scene_id"); sp.add_argument("--label", required=True); sp.add_argument("--notes", default=""); sp.set_defaults(func=cmd_label_candidate)
    sp = sub.add_parser("orbiter-run"); sp.add_argument("project_name"); sp.add_argument("scene_id"); sp.set_defaults(func=cmd_orbiter_run)
    sp = sub.add_parser("orbiter-evaluate"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT); sp.add_argument("--limit", type=int, default=20); sp.set_defaults(func=cmd_orbiter_evaluate)
    sp = sub.add_parser("gemma-status"); sp.set_defaults(func=cmd_gemma_status)
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = ["tui"]
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as exc:
        raise SystemExit(f"Not found: {exc}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON: {exc}") from None
    except csv.Error as exc:
        raise SystemExit(f"Invalid CSV: {exc}") from None
    except curses.error as exc:
        raise SystemExit(f"TUI failed: {exc}. Try 'dsense doctor' to check terminal support.") from None
    except OSError as exc:
        raise SystemExit(f"I/O error: {exc}") from None
    except ValueError as exc:
        raise SystemExit(f"Invalid data: {exc}") from None
