from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
from .baseline import train_and_save_project_baseline
from .baseline_suite import baseline_suite_report_path, run_baseline_suite
from .channels import parse_channel_groups
from .classifier import train_and_save_project_classifier
from .council import intelligence_state_path, load_intelligence_state, run_intelligence_update
from .doctor import doctor_ok, print_doctor_report, run_doctor
from .gemma_edge import gemma_edge_status
from .inputs import validate_capture_params, validate_duration_windows
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
from .scenarios import SCENARIO_GROUPS, Scenario, all_scenarios
from .trace import export_trace, trace_path, viewer_path, write_scene_viewer
from .wizard import guided_scene
from .workloads import valid_workload_ids, workload_progress_callback
from .utils.files import read_json, write_json
from .autotest import validate_dataset, print_validation_report
from .transfer import compare_transfer_bundle, export_transfer_bundle, transfer_bundle_path
from .timeseries import timeseries_path, train_and_save_project_timeseries
from .watcher import label_candidate, run_rolling_watcher, run_watcher_scan


def cmd_init(args):
    root = init_project(args.project_name)
    print(root / "manifest.json"); print(root / "channels.json"); print(root / "scenes"); print(root / "exports")


def cmd_scan(args):
    for ch in scan_channels(advanced=args.advanced):
        status = "available" if ch["available"] else f"unavailable ({ch['reason']})"
        print(f"{ch['id']} [{ch.get('group', 'portable')}]: {status} - {ch['name']}")


def cmd_record_baseline(args):
    validate_capture_params(args.duration, args.tick_hz)
    init_project(args.project_name)
    scene_id = allocate_scene_id(args.project_name)
    groups = parse_channel_groups(args.channels)
    scene = record_scene(project_path(args.project_name) / "scenes" / scene_id, scene_id, "baseline_idle", args.duration, args.tick_hz, 0, args.duration, 0, args.notes, channel_groups=groups)
    print(f"Recorded {scene_id}: confidence={scene['quality']['confidence']}")


def cmd_auto_scenes(args):
    _validate_repeat_tick(args.repeat, args.tick_hz)
    init_project(args.project_name)
    selected = _select_auto_scenarios(args.group, args.include, args.exclude)
    if not selected:
        raise SystemExit("No automatable scenarios selected.")
    total = len(selected) * max(1, args.repeat)
    print(f"Automatic scene batch: project={args.project_name} scenes={len(selected)} repeats={args.repeat} total={total}")
    for scenario in selected:
        workload_text = f" workload={scenario.workload}" if scenario.workload else ""
        print(f"  {scenario.label} [{scenario.mode}]{workload_text} {scenario.duration:g}s")
    if not args.yes:
        answer = input("Start automatic capture batch? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Cancelled.")

    groups = parse_channel_groups(args.channels)
    results: list[dict[str, object]] = []
    for repeat_index in range(1, max(1, args.repeat) + 1):
        for scenario in selected:
            scene_id = allocate_scene_id(args.project_name)
            scene_dir = project_path(args.project_name) / "scenes" / scene_id
            action = scenario.action_seconds
            notes = args.notes or scenario.notes
            callback = workload_progress_callback(scenario.workload, scenario.pre_roll, action)
            print(f"Recording {scene_id} {scenario.label} repeat {repeat_index}/{args.repeat}")
            scene = record_scene(
                scene_dir,
                scene_id,
                scenario.label,
                scenario.duration,
                args.tick_hz,
                scenario.pre_roll,
                action,
                scenario.post_roll,
                notes,
                mode=scenario.mode,
                progress_callback=callback,
                channel_groups=groups,
            )
            scene["accepted"] = True if args.yes else _prompt_keep_scene(scene)
            scene["scenario"] = {
                "mode": scenario.mode,
                "manual": scenario.manual,
                "automatable": scenario.automatable,
                "workload": scenario.workload,
            }
            write_json(scene_dir / "scene.json", scene)
            results.append(scene)
            print(f"Recorded {scene_id}: confidence={scene['quality']['confidence']} accepted={scene['accepted']}")

    _print_auto_scene_training_hint(args.project_name)
    result = validate_dataset(args.project_name)
    print(f"Validation: {result.valid_scenes}/{result.total_scenes} valid, errors={result.error_count}, warnings={result.warning_count}")


def cmd_baseline_suite(args):
    _validate_repeat_tick(args.repeat, args.tick_hz)
    categories = _split_csv(args.categories)
    exclude_categories = _split_csv(args.exclude_categories)
    report = run_baseline_suite(
        args.project_name or DEFAULT_PROJECT,
        target_scenes=args.target_scenes * args.repeat,
        categories=categories or None,
        exclude_categories=exclude_categories or None,
        seed=args.seed,
        duration=args.duration,
        tick_hz=args.tick_hz,
        linux=args.linux,
        dry_run=args.dry_run,
        assume_yes=args.yes,
        include_network=args.include_network,
        include_heavy=args.include_heavy,
    )
    if args.dry_run:
        plan = dict(report["plan"])
        print(f"Baseline suite dry-run: {args.project_name}")
        print(f"planned scenes: {plan['planned_scene_count']} target={plan['target_scenes']}")
        print(f"categories: {', '.join(plan['categories'])}")
        print(f"estimated duration: {plan['estimated_duration_seconds']}s")
        print(f"network: {'enabled' if report.get('network_enabled') else 'disabled'}")
        print(f"linux channels: {'enabled' if args.linux else 'disabled'}")
        for item in list(plan["scenarios"])[: min(20, len(plan["scenarios"]))]:
            print(f"{item['order']:>4} {item['category']:<10} {item['label']} workload={item.get('workload') or 'none'}")
        if int(plan["planned_scene_count"]) > 20:
            print(f"... {int(plan['planned_scene_count']) - 20} more")
        return
    print(f"Baseline suite complete: recorded={report['actual_scene_count']} target={report['target_scene_count']}")
    summary = dict(report.get("validation_summary", {}))
    print(f"Validation: {summary.get('valid_scenes')}/{summary.get('total_scenes')} valid errors={summary.get('errors')} warnings={summary.get('warnings')}")
    print(baseline_suite_report_path(args.project_name or DEFAULT_PROJECT))


def _select_auto_scenarios(group: str, include: str, exclude: str) -> list[Scenario]:
    labels = {scenario.label: scenario for scenario in all_scenarios()}
    include_labels = _split_csv(include)
    exclude_labels = set(_split_csv(exclude))
    if include_labels:
        missing = [label for label in include_labels if label not in labels]
        if missing:
            raise SystemExit(f"Unknown scenario label(s): {', '.join(missing)}")
        candidates = [labels[label] for label in include_labels]
    else:
        group_names = ["baseline", "activity"] if group == "auto" else [group]
        candidates = [scenario for name in group_names for scenario in SCENARIO_GROUPS.get(name, [])]
    selected = [scenario for scenario in candidates if scenario.automatable and scenario.label not in exclude_labels]
    invalid = [scenario.label for scenario in selected if scenario.workload is not None and scenario.workload not in valid_workload_ids()]
    if invalid:
        raise SystemExit(f"Scenario(s) reference unknown workload ids: {', '.join(invalid)}")
    return selected


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _prompt_keep_scene(scene: dict[str, object]) -> bool:
    answer = input(f"Keep {scene['scene_id']} ({scene['label']})? [Y/n]: ").strip().lower()
    return answer not in {"n", "no"}


def _print_auto_scene_training_hint(project_name: str) -> None:
    try:
        baseline_model = train_and_save_project_baseline(project_name)
        classifier_model = train_and_save_project_classifier(project_name)
    except (OSError, ValueError) as exc:
        print(f"Training skipped: {exc}")
        print(f"Run: python -m dsense train-baseline {project_name}")
        print(f"Run: python -m dsense train-classifier {project_name}")
        return
    print(f"Trained baseline: {baseline_model.scene_count} baseline scenes")
    print(f"Trained classifier: {classifier_model.scene_count} scenes")


def _validated_duration(duration: float | None, pre_roll: float, action: float, post_roll: float) -> float:
    return validate_duration_windows(duration, pre_roll, action, post_roll)


def _validate_repeat_tick(repeat: int, tick_hz: int) -> None:
    validate_capture_params(duration=1.0, tick_hz=tick_hz, repeat=repeat)


def _run_tui_config(**kwargs) -> list[dict[str, object]]:
    try:
        from .tui_app import CaptureConfig, run_tui
    except Exception as exc:
        raise SystemExit(f"TUI unavailable: {exc}. Non-TUI commands like doctor, scan, init, scene, and validate are still available.") from None
    try:
        return run_tui(CaptureConfig(**kwargs))
    except Exception as exc:
        if "curses" in exc.__class__.__module__ or exc.__class__.__name__ == "error":
            raise SystemExit(f"TUI failed: {exc}. Try 'dsense doctor' to check terminal support, or use non-TUI commands.") from None
        raise


def _project_scene_stats(project_name: str) -> dict[str, int]:
    root = project_path(project_name) / "scenes"
    stats = {"total": 0, "accepted": 0, "baseline": 0}
    if not root.exists():
        return stats
    for path in sorted(root.glob("scene_*/scene.json")):
        try:
            scene = read_json(path)
        except (OSError, ValueError):
            continue
        stats["total"] += 1
        if scene.get("accepted") is not False:
            stats["accepted"] += 1
            if str(scene.get("label", "")).startswith("baseline_"):
                stats["baseline"] += 1
    return stats


def cmd_scene(args):
    init_project(args.project_name)
    duration = _validated_duration(args.duration, args.pre_roll, args.action, args.post_roll)
    _validate_repeat_tick(args.repeat, args.tick_hz)
    groups = parse_channel_groups(args.channels)
    if args.tui:
        _run_tui_config(
            project_name=args.project_name,
            channel_groups=groups,
            label=args.label,
            duration=duration,
            pre_roll=args.pre_roll,
            action=args.action,
            post_roll=args.post_roll,
            repeat=args.repeat,
            tick_hz=args.tick_hz,
            notes=args.notes,
            auto_baseline_policy="off",
            startup_suite_enabled=False,
        )
        return
    guided_scene(args.project_name, args.label, duration, args.pre_roll, args.action, args.post_roll, args.repeat, args.notes, args.tick_hz, args.yes, channel_groups=groups)


def cmd_tui(args):
    project_name = args.project_name or DEFAULT_PROJECT
    groups = parse_channel_groups(args.channels)
    print("Opening dSense...")
    print(f"Project: {project_name}")
    policy = "off" if args.no_auto_baseline else args.auto_baseline_policy
    startup_intelligence = not args.no_startup_intelligence
    if not startup_intelligence:
        policy = "off"
        args.no_startup_suite = True
        args.no_startup_watchers = True
        args.no_startup_orbiters = True
        args.no_startup_training = True
        print("Startup intelligence disabled for this session.")
    print("Opening TUI startup pipeline..." if startup_intelligence else "Opening TUI without startup intelligence...")
    duration = _validated_duration(args.duration, args.pre_roll, args.action, args.post_roll)
    _validate_repeat_tick(args.repeat, args.tick_hz)
    _run_tui_config(
        project_name=project_name,
        channel_groups=groups,
        label=args.label,
        duration=duration,
        pre_roll=args.pre_roll,
        action=args.action,
        post_roll=args.post_roll,
        repeat=args.repeat,
        tick_hz=args.tick_hz,
        notes=args.notes,
        auto_baseline_policy=policy,
        auto_baseline_duration=args.auto_baseline_duration,
        force_auto_baseline=args.force_auto_baseline,
        startup_suite_enabled=not args.no_startup_suite,
        startup_suite_target=args.startup_suite_target,
        startup_suite_duration=args.startup_suite_duration,
        startup_suite_seed=args.startup_suite_seed,
        startup_suite_linux=args.startup_suite_linux,
        startup_intelligence=startup_intelligence,
        startup_watchers=not args.no_startup_watchers,
        startup_orbiters=not args.no_startup_orbiters,
        startup_training=not args.no_startup_training,
    )


def cmd_tui_safe(args):
    args.no_startup_intelligence = True
    args.no_startup_watchers = True
    args.no_startup_orbiters = True
    args.no_startup_training = True
    args.no_auto_baseline = True
    args.no_startup_suite = True
    cmd_tui(args)


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
    print(f"Validating dataset: {args.project_name}", flush=True)
    result = validate_dataset(args.project_name)
    print_validation_report(result, verbose=args.verbose)
    print(f"Validation complete: {result.valid_scenes}/{result.total_scenes} valid, errors={result.error_count}, warnings={result.warning_count}")


def cmd_doctor(args):
    checks = run_doctor(args.project_name)
    print_doctor_report(checks)
    if not doctor_ok(checks):
        raise SystemExit(1)


def cmd_train_classifier(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    stats = _project_scene_stats(project_name)
    print(f"Training classifier: {project_name}", flush=True)
    print(f"  scenes discovered: {stats['total']} total, {stats['accepted']} accepted, {stats['baseline']} baseline labels", flush=True)
    _require_valid_dataset(project_name, args.require_valid)
    print("  extracting preview features and label profiles...", flush=True)
    model = train_and_save_project_classifier(project_name)
    manifest = dict(model.feature_manifest)
    print(f"  model labels: {len(model.label_counts)} labels, {manifest.get('feature_count', 0)} features", flush=True)
    print(f"Trained classifier for {project_name}: {model.scene_count} scenes, {model.baseline_scene_count} baseline scenes")
    print(project_path(project_name) / "exports" / "classifier.json")


def cmd_train_baseline(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    stats = _project_scene_stats(project_name)
    print(f"Training baseline: {project_name}", flush=True)
    print(f"  scenes discovered: {stats['total']} total, {stats['accepted']} accepted, {stats['baseline']} baseline labels", flush=True)
    _require_valid_dataset(project_name, args.require_valid)
    print("  reading baseline previews and computing robust channel profiles...", flush=True)
    model = train_and_save_project_baseline(project_name)
    manifest = dict(model.feature_manifest)
    print(f"  model channels: {len(model.channels)} channels, {manifest.get('feature_count', 0)} features", flush=True)
    print(f"Trained baseline for {project_name}: {model.scene_count} baseline scenes, {len(model.channels)} channels")
    print(project_path(project_name) / "exports" / "baseline_model.json")


def cmd_train_timeseries(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    stats = _project_scene_stats(project_name)
    print(f"Training time-series model: {project_name}", flush=True)
    print(f"  scenes discovered: {stats['total']} total, {stats['accepted']} accepted, {stats['baseline']} baseline labels", flush=True)
    _require_valid_dataset(project_name, args.require_valid)
    print("  extracting temporal preview features and label profiles...", flush=True)
    model = train_and_save_project_timeseries(project_name)
    manifest = dict(model.feature_manifest)
    print(f"  model labels: {len(model.label_counts)} labels, {manifest.get('feature_count', 0)} features", flush=True)
    print(f"Trained time-series model for {project_name}: {model.scene_count} scenes, {len(model.sequence_channels)} sequence channels")
    print(timeseries_path(project_name))


def cmd_update_intelligence(args):
    project_name = args.project_name or DEFAULT_PROJECT
    init_project(project_name)
    print(f"Updating local intelligence stack: {project_name}", flush=True)

    def progress(update: dict[str, object]) -> None:
        step = dict(update.get("step", {}))
        status = str(step.get("status", ""))
        name = str(step.get("name", ""))
        summary = dict(step.get("summary", {})) if isinstance(step.get("summary"), dict) else {}
        detail = summary.get("error") or summary.get("path") or summary.get("scene_count") or ""
        print(f"  {name}: {status}{f' ({detail})' if detail != '' else ''}", flush=True)

    state = run_intelligence_update(
        project_name,
        startup=False,
        run_watchers=not args.no_watchers,
        run_orbiters=not args.no_orbiters,
        run_training=not args.no_training,
        run_transfer=not args.no_transfer,
        progress_callback=progress,
    )
    council = dict(state.get("council", {}))
    print(f"Status: {state.get('status')}")
    print(f"Agreement: {council.get('agreement')}  confidence={council.get('overall_confidence')}")
    print(intelligence_state_path(project_name))


def cmd_council_status(args):
    project_name = args.project_name or DEFAULT_PROJECT
    state = load_intelligence_state(project_name)
    if state is None:
        raise SystemExit(f"No intelligence state found. Run: python -m dsense update-intelligence {project_name}")
    council = dict(state.get("council", {}))
    models = dict(state.get("models", {}))
    print(f"Intelligence Council: {project_name}")
    print(f"Status: {state.get('status')}  Agreement: {council.get('agreement')}  Confidence: {council.get('overall_confidence')}")
    for name in ("baseline", "classifier", "timeseries", "watcher", "orbiters", "evaluation", "transfer"):
        print(f"{name}: {models.get(name, {})}")
    warnings = list(council.get("warnings", []))
    recommendations = list(council.get("recommendations", []))
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")
    if recommendations:
        print("Recommendations:")
        for item in recommendations:
            print(f"  - {item}")
    print(intelligence_state_path(project_name))


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
    validate_capture_params(max(args.duration, 0.01) if args.duration else 0.01, args.tick_hz)
    groups = parse_channel_groups(args.channels)
    if args.rolling:
        print(f"Starting rolling watcher: project={project_name} tick_hz={args.tick_hz} channels={','.join(groups)} pre={args.pre}s post={args.post}s duration={args.duration or 'continuous'}", flush=True)
        result = run_rolling_watcher(
            project_name,
            pre_seconds=args.pre,
            post_seconds=args.post,
            tick_hz=args.tick_hz,
            cooldown_seconds=args.cooldown,
            duration=args.duration,
            channel_groups=groups,
            prompt_label=args.prompt_label,
        )
        print(f"Rolling watcher saved {len(result['saved'])} anomaly windows")
        print(result["session_path"])
        for saved in result["saved"]:
            scene = dict(saved["scene"])
            event = dict(saved["event"])
            print(f"{scene.get('scene_id')} score={event.get('anomaly_score')} label={scene.get('label')} accepted={scene.get('accepted')}")
        return
    print(f"Starting watcher scan: project={project_name} duration={args.duration or 5.0}s tick_hz={args.tick_hz} channels={','.join(groups)}", flush=True)
    result = run_watcher_scan(project_name, duration=args.duration or 5.0, tick_hz=args.tick_hz, channel_groups=groups)
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


def _add_tui_args(sp):
    sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to open (default: {DEFAULT_PROJECT})")
    sp.add_argument("--label", default="user_interaction")
    sp.add_argument("--duration", type=float)
    sp.add_argument("--pre-roll", type=float, default=2)
    sp.add_argument("--action", type=float, default=5)
    sp.add_argument("--post-roll", type=float, default=3)
    sp.add_argument("--repeat", type=int, default=1)
    sp.add_argument("--notes", default="")
    sp.add_argument("--tick-hz", type=int, default=100)
    sp.add_argument("--channels", default="portable", help="channel groups, e.g. portable or portable,linux")
    sp.add_argument("--auto-baseline-policy", choices=["auto", "startup", "missing-only", "off"], default="auto", help="startup baseline policy; disabled by --no-startup-intelligence and tui-safe")
    sp.add_argument("--no-auto-baseline", action="store_true", help="same as --auto-baseline-policy off")
    sp.add_argument("--auto-baseline-duration", type=float, default=5.0, help="seconds for automatic startup baseline capture")
    sp.add_argument("--force-auto-baseline", action="store_true", help="record a fresh startup baseline unless policy is off")
    sp.add_argument("--no-startup-suite", action="store_true", help="skip filling the automatic system baseline/control suite on TUI startup")
    sp.add_argument("--startup-suite-target", type=int, default=200, help="target number of baseline-suite scenes to maintain before opening TUI")
    sp.add_argument("--startup-suite-duration", type=float, default=0.2, help="seconds per automatic startup-suite scene")
    sp.add_argument("--startup-suite-seed", type=int, default=42)
    sp.add_argument("--startup-suite-linux", action=argparse.BooleanOptionalAction, default=True, help="include Linux-safe startup-suite controls")
    sp.add_argument("--no-startup-intelligence", action="store_true", help="open without automatic intelligence update, watcher scan, orbiter run, training, or startup suite")
    sp.add_argument("--no-startup-watchers", action="store_true", help="skip startup watcher scan inside the intelligence update")
    sp.add_argument("--no-startup-orbiters", action="store_true", help="skip startup orbiter evaluation inside the intelligence update")
    sp.add_argument("--no-startup-training", action="store_true", help="load existing models instead of retraining during startup intelligence")


def build_parser():
    p = argparse.ArgumentParser(prog="dsense", description="dSense Scene Wizard")
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("init"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_init)
    sp = sub.add_parser("doctor"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to check (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_doctor)
    sp = sub.add_parser("scan"); sp.add_argument("--advanced", action="store_true", help="include linux and experimental adapters"); sp.set_defaults(func=cmd_scan)
    sp = sub.add_parser("record-baseline"); sp.add_argument("project_name"); sp.add_argument("--duration", type=float, default=30); sp.add_argument("--tick-hz", type=int, default=100); sp.add_argument("--notes", default=""); sp.add_argument("--channels", default="portable", help="channel groups, e.g. portable or portable,linux"); sp.set_defaults(func=cmd_record_baseline)
    sp = sub.add_parser("auto-scenes")
    sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to record into (default: {DEFAULT_PROJECT})")
    sp.add_argument("--group", choices=["auto", "baseline", "activity"], default="auto", help="automatable preset group to record")
    sp.add_argument("--repeat", type=int, default=1, help="number of repeats per selected scenario")
    sp.add_argument("--include", default="", help="comma-separated scenario labels to include")
    sp.add_argument("--exclude", default="", help="comma-separated scenario labels to exclude")
    sp.add_argument("--tick-hz", type=int, default=100)
    sp.add_argument("--channels", default="portable", help="channel groups, e.g. portable or portable,linux")
    sp.add_argument("--notes", default="", help="override scenario notes for this batch")
    sp.add_argument("--yes", action="store_true", help="start and accept captures without prompts")
    sp.set_defaults(func=cmd_auto_scenes)
    sp = sub.add_parser("baseline-suite")
    sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to record into (default: {DEFAULT_PROJECT})")
    sp.add_argument("--linux", action=argparse.BooleanOptionalAction, default=True, help="include Linux channel group and Linux-safe proc/sysfs controls")
    sp.add_argument("--target-scenes", type=int, default=200)
    sp.add_argument("--repeat", type=int, default=1, help="multiply the target scene plan by this repeat count")
    sp.add_argument("--categories", default="", help="comma-separated categories to include")
    sp.add_argument("--exclude-categories", default="", help="comma-separated categories to exclude")
    sp.add_argument("--seed", type=int)
    sp.add_argument("--duration", type=float, default=1.0, help="seconds per planned scene")
    sp.add_argument("--tick-hz", type=int, default=50)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--yes", action="store_true", help="run unattended")
    sp.add_argument("--include-network", action="store_true", help="include network controls only when DSENSE_NET_HOST is configured")
    sp.add_argument("--include-heavy", action="store_true", help="include heavier opt-in workloads")
    sp.set_defaults(func=cmd_baseline_suite)
    sp = sub.add_parser("scene"); sp.add_argument("project_name"); sp.add_argument("--label", required=True); sp.add_argument("--duration", type=float); sp.add_argument("--pre-roll", type=float, default=2); sp.add_argument("--action", type=float, default=5); sp.add_argument("--post-roll", type=float, default=3); sp.add_argument("--repeat", type=int, default=1); sp.add_argument("--notes", default=""); sp.add_argument("--tick-hz", type=int, default=100); sp.add_argument("--channels", default="portable", help="channel groups, e.g. portable or portable,linux"); sp.add_argument("--yes", action="store_true", help="accept captures without prompt"); sp.add_argument("--tui", action="store_true", help="record with the full-screen interaction recorder"); sp.set_defaults(func=cmd_scene)
    sp = sub.add_parser("tui"); _add_tui_args(sp); sp.set_defaults(func=cmd_tui)
    sp = sub.add_parser("tui-safe", help="open the TUI without startup intelligence, watchers, orbiters, training, or baseline-suite work"); _add_tui_args(sp); sp.set_defaults(func=cmd_tui_safe)
    sp = sub.add_parser("list-scenes"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_list)
    sp = sub.add_parser("export-preview"); sp.add_argument("project_name"); sp.set_defaults(func=cmd_export)
    sp = sub.add_parser("validate"); sp.add_argument("project_name"); sp.add_argument("--verbose", "-v", action="store_true", help="show detailed error messages"); sp.set_defaults(func=cmd_validate)
    sp = sub.add_parser("train-baseline"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to train (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_train_baseline)
    sp.add_argument("--require-valid", action="store_true", help="fail before training if dataset validation has errors")
    sp = sub.add_parser("train-classifier"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to train (default: {DEFAULT_PROJECT})"); sp.add_argument("--require-valid", action="store_true", help="fail before training if dataset validation has errors"); sp.set_defaults(func=cmd_train_classifier)
    sp = sub.add_parser("train-timeseries"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to train (default: {DEFAULT_PROJECT})"); sp.add_argument("--require-valid", action="store_true", help="fail before training if dataset validation has errors"); sp.set_defaults(func=cmd_train_timeseries)
    sp = sub.add_parser("update-intelligence"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to update (default: {DEFAULT_PROJECT})"); sp.add_argument("--no-watchers", action="store_true", help="skip watcher scan and use existing watcher events"); sp.add_argument("--no-orbiters", action="store_true", help="skip orbiter evaluation and use existing summaries"); sp.add_argument("--no-training", action="store_true", help="load existing models instead of retraining"); sp.add_argument("--no-transfer", action="store_true", help="skip transfer bundle export"); sp.set_defaults(func=cmd_update_intelligence)
    sp = sub.add_parser("council-status"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to inspect (default: {DEFAULT_PROJECT})"); sp.set_defaults(func=cmd_council_status)
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
    sp = sub.add_parser("watcher"); sp.add_argument("project_name", nargs="?", default=DEFAULT_PROJECT, help=f"project to watch (default: {DEFAULT_PROJECT})"); sp.add_argument("--rolling", action="store_true", help="use rolling anomaly windows instead of scan recording"); sp.add_argument("--pre", type=float, default=5.0, help="seconds to keep before trigger"); sp.add_argument("--post", type=float, default=10.0, help="seconds to save after trigger"); sp.add_argument("--cooldown", type=float, default=30.0, help="seconds before saving another window"); sp.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 means indefinitely in rolling mode"); sp.add_argument("--tick-hz", type=int, default=50); sp.add_argument("--channels", default="portable", help="channel groups, e.g. portable or portable,linux"); sp.add_argument("--prompt-label", action="store_true", help="prompt for a label when an anomaly is saved"); sp.set_defaults(func=cmd_watcher)
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
    except OSError as exc:
        raise SystemExit(f"I/O error: {exc}") from None
    except ValueError as exc:
        raise SystemExit(f"Invalid data: {exc}") from None
