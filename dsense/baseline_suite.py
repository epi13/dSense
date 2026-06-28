from __future__ import annotations

import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path

from .autotest import validate_dataset
from .baseline import train_and_save_project_baseline
from .classifier import train_and_save_project_classifier
from .inputs import validate_capture_params
from .manifest import allocate_scene_id, init_project, project_path, scan_channels
from .models.evaluation import evaluate_project_scenes
from .recorder import record_scene
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso
from .workloads import valid_workload_ids, workload_progress_callback


SUITE_REPORT_NAME = "baseline_suite_report.json"
DEFAULT_CATEGORIES = ("idle", "timing", "cpu", "memory", "disk", "proc", "combined", "drift")


@dataclass(frozen=True)
class BaselineSuiteScenario:
    category: str
    name: str
    workload: str | None = None
    duration_scale: float = 1.0
    linux_only: bool = False
    network: bool = False
    heavy: bool = False

    @property
    def base_label(self) -> str:
        return f"baseline_{self.category}_{self.name}"


def baseline_suite_report_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / SUITE_REPORT_NAME


def baseline_suite_catalog(include_network: bool = False, include_heavy: bool = False) -> list[BaselineSuiteScenario]:
    scenarios: list[BaselineSuiteScenario] = []

    def add(category: str, names: list[str], workload: str | None = None, **kwargs: object) -> None:
        for name in names:
            scenarios.append(BaselineSuiteScenario(category, name, workload=workload, **kwargs))

    add("idle", ["quiet_short", "quiet_medium", "warmed_idle", "post_startup_idle", "screen_on_idle", "terminal_idle", "quiet_control_a", "quiet_control_b", "cooldown_idle"], "noop")
    add("timing", ["pure_sleep", "short_interval", "long_interval", "jitter_observe", "clock_delta_focus", "process_probe_focus", "scheduler_quiet"], "noop")
    add("cpu", ["no_load", "light_loop", "bursty_loop", "modest_loop", "single_thread_python", "mixed_sleep_cpu", "cooldown_after_cpu"], "cpu_light")
    add("memory", ["no_allocation", "small_allocation", "medium_allocation", "allocate_release", "list_dict_create", "string_allocate", "cooldown_after_memory"], "memory_allocate_release")
    add("disk", ["cwd_stat_only", "temp_create_delete", "metadata_writes", "small_sequential_write", "small_sequential_read", "limited_fsync_control", "cooldown_after_disk"], "disk_stat_burst")
    add("proc", ["proc_stat_read", "proc_self_status_read", "proc_meminfo_read", "thermal_sysfs_read", "power_sysfs_read", "proc_sleep_mix"], "proc_read", linux_only=True)
    add("combined", ["light_cpu_disk", "memory_cpu", "proc_reads_sleep", "disk_metadata_cpu", "mixed_tiny_workload", "cooldown_after_mixed"], "mixed_cpu_disk")
    add("drift", ["idle_begin", "idle_middle", "idle_end", "cpu_repeat_begin", "cpu_repeat_middle", "cpu_repeat_end", "disk_repeat_begin", "disk_repeat_middle", "disk_repeat_end"], "noop")
    if include_network:
        add("network", ["disabled_baseline", "connect_latency", "failure_degraded"], "noop", network=True)
    if include_heavy:
        add("cpu", ["heavy_opt_in"], "cpu_heavy", heavy=True)
    return scenarios


def plan_baseline_suite(
    target_scenes: int = 200,
    categories: list[str] | tuple[str, ...] | None = None,
    exclude_categories: list[str] | tuple[str, ...] | None = None,
    seed: int | None = None,
    duration: float = 1.0,
    include_network: bool = False,
    include_heavy: bool = False,
    linux: bool = True,
    label_offset: int = 0,
) -> dict[str, object]:
    if target_scenes < 1:
        raise ValueError("target_scenes must be >= 1")
    if duration <= 0:
        raise ValueError("duration must be positive")
    selected_categories = set(categories or DEFAULT_CATEGORIES)
    excluded = set(exclude_categories or ())
    catalog = [
        scenario
        for scenario in baseline_suite_catalog(include_network=include_network, include_heavy=include_heavy)
        if scenario.category in selected_categories and scenario.category not in excluded
        and (linux or not scenario.linux_only)
        and (include_network or not scenario.network)
        and (include_heavy or not scenario.heavy)
    ]
    if not catalog:
        raise ValueError("No baseline suite scenarios selected")
    for scenario in catalog:
        if scenario.workload is not None and scenario.workload not in valid_workload_ids():
            raise ValueError(f"Unknown workload id in baseline suite: {scenario.workload}")

    rng = random.Random(seed)
    order: list[BaselineSuiteScenario] = []
    cycle = 0
    while len(order) < target_scenes:
        shuffled = list(catalog)
        rng.shuffle(shuffled)
        if cycle % 3 == 1:
            drift = [scenario for scenario in catalog if scenario.category == "drift"]
            rng.shuffle(drift)
            shuffled = drift + [scenario for scenario in shuffled if scenario.category != "drift"]
        order.extend(shuffled)
        cycle += 1
    order = order[:target_scenes]
    planned = []
    for index, scenario in enumerate(order, start=1):
        label_index = label_offset + index
        planned.append({
            "order": index,
            "category": scenario.category,
            "name": scenario.name,
            "base_label": scenario.base_label,
            "label": f"{scenario.base_label}_{label_index:03d}",
            "workload": scenario.workload,
            "duration": round(duration * scenario.duration_scale, 6),
            "manual": False,
            "negative_control": True,
        })
    category_counts: dict[str, int] = {}
    for item in planned:
        category = str(item["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "format": "dsense-baseline-suite-plan-v1",
        "target_scenes": target_scenes,
        "planned_scene_count": len(planned),
        "categories": sorted(category_counts),
        "category_counts": category_counts,
        "seed": seed,
        "duration": duration,
        "estimated_duration_seconds": round(sum(float(item["duration"]) for item in planned), 3),
        "network_enabled": include_network,
        "heavy_enabled": include_heavy,
        "scenarios": planned,
    }


def run_baseline_suite(
    project_name: str,
    target_scenes: int = 200,
    categories: list[str] | tuple[str, ...] | None = None,
    exclude_categories: list[str] | tuple[str, ...] | None = None,
    seed: int | None = None,
    duration: float = 1.0,
    tick_hz: int = 50,
    linux: bool = True,
    dry_run: bool = False,
    assume_yes: bool = False,
    include_network: bool = False,
    include_heavy: bool = False,
    label_offset: int = 0,
) -> dict[str, object]:
    validate_capture_params(duration, tick_hz)
    init_project(project_name)
    network_enabled = include_network and bool(os.environ.get("DSENSE_NET_HOST"))
    plan = plan_baseline_suite(target_scenes, categories, exclude_categories, seed, duration, network_enabled, include_heavy, linux, label_offset=label_offset)
    channel_groups = ["portable", "linux"] if linux else ["portable"]
    channel_status = scan_channels(advanced=linux, groups=channel_groups)
    if dry_run:
        return {
            "format": "dsense-baseline-suite-report-v1",
            "dry_run": True,
            "project_name": project_name,
            "plan": plan,
            "channel_groups": channel_groups,
            "channels": channel_status,
            "network_requested": include_network,
            "network_enabled": network_enabled,
        }
    if not assume_yes:
        raise SystemExit("Refusing to run unattended baseline suite without --yes")

    suite_id = f"suite_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}_{uuid.uuid4().hex[:8]}"
    recorded: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for item in list(plan["scenarios"]):
        try:
            scene_id = allocate_scene_id(project_name)
            scene_dir = project_path(project_name) / "scenes" / scene_id
            item_duration = float(item["duration"])
            workload = item.get("workload")
            callback = workload_progress_callback(str(workload) if workload else None, 0.0, item_duration)
            scene = record_scene(
                scene_dir,
                scene_id,
                str(item["label"]),
                item_duration,
                tick_hz,
                0.0,
                item_duration,
                0.0,
                "Automatic baseline-suite negative control; no intentional physical interaction.",
                mode="baseline_suite",
                progress_callback=callback,
                channel_groups=channel_groups,
            )
            scene["suite"] = {"suite_id": suite_id, **item}
            scene["accepted"] = True
            write_json(scene_dir / "scene.json", scene)
            recorded.append({"scene_id": scene_id, **item})
        except Exception as exc:
            failed.append({"scenario": item, "error": str(exc)})

    baseline_model = train_and_save_project_baseline(project_name)
    classifier_model = train_and_save_project_classifier(project_name)
    validation = validate_dataset(project_name)
    evaluation = evaluate_project_scenes(project_name)
    report = {
        "format": "dsense-baseline-suite-report-v1",
        "dry_run": False,
        "suite_id": suite_id,
        "created_utc": utc_now_iso(),
        "project_name": project_name,
        "target_scene_count": target_scenes,
        "actual_scene_count": len(recorded),
        "categories_included": plan["categories"],
        "labels_generated": [str(item["label"]) for item in plan["scenarios"]],
        "scenario_order": recorded,
        "seed": seed,
        "duration": duration,
        "tick_hz": tick_hz,
        "channel_groups": channel_groups,
        "channels": channel_status,
        "failed_or_skipped": failed,
        "validation_summary": {
            "total_scenes": validation.total_scenes,
            "valid_scenes": validation.valid_scenes,
            "errors": validation.error_count,
            "warnings": validation.warning_count,
        },
        "baseline_model_summary": {
            "scene_count": baseline_model.scene_count,
            "channel_count": len(baseline_model.channels),
            "feature_count": int(dict(baseline_model.feature_manifest).get("feature_count", 0)),
        },
        "classifier_summary": {
            "scene_count": classifier_model.scene_count,
            "baseline_scene_count": classifier_model.baseline_scene_count,
            "label_count": len(classifier_model.label_counts),
        },
        "drift_summary": evaluation.get("baseline_drift", {}),
        "noisy_channel_summary": noisy_channel_summary(baseline_model.channels),
    }
    out = baseline_suite_report_path(project_name)
    ensure_dir(out.parent)
    write_json(out, report)
    return report


def noisy_channel_summary(channels: dict[str, dict[str, float]], limit: int = 10) -> list[dict[str, object]]:
    rows = []
    for channel, profile in channels.items():
        center = abs(float(profile.get("center", 0.0)))
        mad = abs(float(profile.get("mad", 0.0)))
        variance = abs(float(profile.get("variance", 0.0)))
        score = mad / max(center, 1.0) + variance / max(center * center, 1.0)
        rows.append({"channel": channel, "instability_score": round(score, 6), "mad": mad, "variance": variance})
    return sorted(rows, key=lambda item: (-float(item["instability_score"]), str(item["channel"])))[:limit]


def count_baseline_suite_scenes(project_name: str) -> int:
    scenes_root = project_path(project_name) / "scenes"
    if not scenes_root.exists():
        return 0
    count = 0
    for scene_path in sorted(scenes_root.glob("scene_*/scene.json")):
        try:
            scene = read_json(scene_path)
        except (OSError, ValueError):
            continue
        suite = scene.get("suite")
        if scene.get("accepted") is not False and (scene.get("mode") == "baseline_suite" or isinstance(suite, dict)):
            count += 1
    return count


def ensure_startup_baseline_suite(
    project_name: str,
    target_scenes: int = 200,
    duration: float = 0.2,
    tick_hz: int = 50,
    linux: bool = True,
    seed: int | None = 42,
    enabled: bool = True,
) -> dict[str, object]:
    if target_scenes < 1:
        raise ValueError("target_scenes must be >= 1")
    validate_capture_params(duration, tick_hz)
    init_project(project_name)
    if not enabled:
        return {"status": "skipped", "recorded": 0, "message": "Startup system suite: skipped"}
    existing = count_baseline_suite_scenes(project_name)
    if existing >= target_scenes:
        return {
            "status": "reused",
            "recorded": 0,
            "existing": existing,
            "target": target_scenes,
            "message": f"Startup system suite: already has {existing}/{target_scenes} suite scenes",
        }
    missing = target_scenes - existing
    report = run_baseline_suite(
        project_name,
        target_scenes=missing,
        seed=seed,
        duration=duration,
        tick_hz=tick_hz,
        linux=linux,
        assume_yes=True,
        label_offset=existing,
    )
    recorded = int(report.get("actual_scene_count", 0))
    return {
        "status": "recorded",
        "recorded": recorded,
        "existing": existing,
        "target": target_scenes,
        "message": f"Startup system suite: recorded {recorded} scenes ({existing + recorded}/{target_scenes})",
        "report": report,
    }
