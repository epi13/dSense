from __future__ import annotations

import curses
import queue
import threading
import time
from collections import deque

from .baseline import BaselineModel, default_auto_baseline_policy, load_project_baseline, project_has_usable_baseline
from .baseline_suite import count_baseline_suite_scenes, run_baseline_suite
from .classifier import SceneClassifierModel, load_project_classifier
from .council import load_intelligence_state, run_intelligence_update
from .doctor import doctor_ok, run_doctor
from .event_detector import DetectorState, HeuristicEventDetector
from .gemma_edge import gemma_edge_status
from .manifest import allocate_scene_id, init_project, project_path, scan_channels
from .orbiters import read_recent_orbiter_summaries
from .recorder import record_scene
from .scenarios import SCENARIO_GROUPS
from .startup_progress import OPTIONAL_STARTUP_STEPS, initial_progress_rows, make_progress, progress_warning
from .transfer import transfer_bundle_path
from .timeseries import TimeSeriesModel, load_project_timeseries
from .tui_live import (
    LiveObservation,
    LiveSampler,
    LiveSessionWriter,
    build_live_observation,
    save_live_snapshot,
)
from .tui_jobs import TuiJobManager, evaluate_project_job, export_transfer_job, run_watcher_job, train_baseline, train_classifier, update_intelligence_job, validate_project
from .tui_render import (
    MIN_TUI_HEIGHT,
    MIN_TUI_WIDTH,
    TABS,
    channel_state_label,
    classifier_summary_lines,
    clip_text,
    compact_live_observation_lines,
    council_summary_lines,
    evaluation_repeatability_lines,
    format_metric_value,
    labels_needing_more_takes,
    live_observation_lines,
    profile_line,
    robust_channel_score,
    scene_detail_lines,
    sense_radar_lines,
    scheduled_scene_events,
    summarize_scene_counts,
    sparkline,
    system_event_marker,
    tab_index_delta,
    useful_channel_lines,
    value_channel_id,
    wrap_text,
)
from .tui_state import CaptureConfig
from .utils.files import read_json, write_json
from .watcher import read_recent_watcher_events
from .workloads import workload_progress_callback


class SceneRecorderTUI:
    STATUS_PANEL_HEIGHT = 5

    def __init__(self, screen, config: CaptureConfig):
        self.screen = screen
        self.config = config
        self.channels = scan_channels(groups=self.config.channel_groups)
        self.scenes = load_project_scenes(config.project_name)
        self.baseline: BaselineModel | None = None
        self.classifier: SceneClassifierModel | None = None
        self.timeseries: TimeSeriesModel | None = None
        self.intelligence_state: dict[str, object] | None = load_intelligence_state(config.project_name)
        self.tab_index = self._initial_tab_index(config.start_tab)
        self.phase_index = 0
        self.scene_index = max(0, len(self.scenes) - 1)
        self.scene_scroll = 0
        self.last_validation_result = None
        self.evaluation_report: dict[str, object] | None = None
        self.job_manager = TuiJobManager(config.project_name)
        self.validation_summary = "not run"
        self.watcher_summary = "idle"
        self.transfer_summary = "not exported"
        self.messages: list[str] = []
        self.events: list[dict[str, object]] = []
        self.live_events: list[dict[str, object]] = []
        self.latest: dict[str, object] = {}
        self.channel_history: dict[str, deque[float]] = {}
        self.detector_state = DetectorState()
        self.last_draw = 0.0
        self.live_sampler: LiveSampler | None = None
        self.live_writer = LiveSessionWriter(config.project_name)
        self.live_observation: LiveObservation | None = None
        self.live_rows: deque[dict[str, float]] = deque(maxlen=max(8, int(config.tick_hz)))
        self.live_started_monotonic = time.monotonic()
        self.live_message = ""
        self._apply_selected_scenario()

    def _initial_tab_index(self, start_tab: str) -> int:
        requested = {"capture": "Capture", "live": "Live", "radar": "Sense Radar", "sense-radar": "Sense Radar"}.get(start_tab.lower(), start_tab.title())
        return TABS.index(requested) if requested in TABS else 0

    def _train_classifier(self) -> SceneClassifierModel | None:
        return train_classifier(self.config.project_name)

    def _train_baseline(self) -> BaselineModel | None:
        return train_baseline(self.config.project_name)

    def _load_classifier(self) -> SceneClassifierModel | None:
        return load_project_classifier(self.config.project_name)

    def run(self) -> list[dict[str, object]]:
        curses.curs_set(0)
        self.screen.nodelay(False)
        self.screen.keypad(True)
        self.screen.timeout(250)
        self._setup_colors()
        self._run_startup_pipeline()
        try:
            return self._main_loop()
        finally:
            self._close_live_sampler()

    def _main_loop(self) -> list[dict[str, object]]:
        all_results = []
        while True:
            action = self._configure()
            if action == "quit":
                return all_results
            if action == "record":
                self._confirm_ready()
                results = self._record_session()
                all_results.extend(results)
                self._complete(results)

    def _record_session(self) -> list[dict[str, object]]:
        results = []
        take = 1
        queue = self._capture_queue()
        while take <= len(queue):
            scenario = queue[take - 1]
            if scenario is not None:
                self._apply_scenario(scenario)
            self.events = []
            self.live_events = []
            self.latest = {}
            self.channel_history = {}
            self.messages = [
                f"Take {take}/{len(queue)}",
                "Press SPACE to mark an interaction. Press n to mark noise. Press q to flag review.",
            ]
            self._countdown()
            scene = self._record_take(take)
            decision = self._review(scene)
            scene["accepted"] = decision == "keep"
            write_json(project_path(self.config.project_name) / "scenes" / scene["scene_id"] / "scene.json", scene)
            results.append(scene)
            self._upsert_scene(scene)
            if decision == "retake":
                queue.insert(take, scenario)
            take += 1
        return results

    def _upsert_scene(self, scene: dict[str, object]) -> None:
        scene_id = str(scene.get("scene_id", ""))
        for index, existing in enumerate(self.scenes):
            if str(existing.get("scene_id", "")) == scene_id:
                self.scenes[index] = scene
                self.scene_index = index
                return
        self.scenes.append(scene)
        self.scene_index = len(self.scenes) - 1

    def _setup_colors(self) -> None:
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)

    def _run_startup_pipeline(self) -> None:
        if not self.config.startup_intelligence:
            init_project(self.config.project_name)
            self.channels = scan_channels(groups=self.config.channel_groups)
            self.scenes = load_project_scenes(self.config.project_name)
            self.baseline = load_project_baseline(self.config.project_name)
            self.classifier = self._load_classifier()
            self.timeseries = load_project_timeseries(self.config.project_name)
            self.config.startup_baseline_status = "Startup intelligence disabled for this session."
            self.messages.append(self.config.startup_baseline_status)
            self._draw_startup_pipeline([
                {"key": "disabled", "label": "Startup intelligence disabled for this session.", "status": "skipped", "detail": "existing artifacts loaded"},
                {"key": "dashboard", "label": "open dashboard", "status": "done", "detail": "ready"},
            ])
            time.sleep(0.4)
            return

        if self.config.startup_mode in {"balanced", "fast"} and not self.config.force_startup_update:
            self._run_open_first_startup_pipeline()
            return

        self._run_council_startup_pipeline()

    def _run_open_first_startup_pipeline(self) -> None:
        init_project(self.config.project_name)
        self.channels = scan_channels(groups=self.config.channel_groups)
        self.scenes = load_project_scenes(self.config.project_name)
        self.baseline = load_project_baseline(self.config.project_name)
        self.classifier = self._load_classifier()
        self.timeseries = load_project_timeseries(self.config.project_name)
        self.intelligence_state = load_intelligence_state(self.config.project_name)
        status = "Fast start: existing artifacts loaded." if self.config.startup_mode == "fast" else "Live opened with cached artifacts."
        self.config.startup_baseline_status = status
        self.messages.append(status)
        rows = [
            {"key": "load", "label": "load cached artifacts", "status": "done", "detail": "ready"},
            {"key": "dashboard", "label": "open live observatory", "status": "done", "detail": "ready"},
        ]
        if self.config.startup_mode == "balanced":
            self._start_background_startup_refresh()
            rows.append({"key": "refresh", "label": "background intelligence refresh", "status": "active", "detail": "cache-aware"})
            self.live_message = "intelligence refresh running in background; press u for a full refresh"
        self._draw_startup_pipeline(rows)
        time.sleep(0.25)

    def _start_background_startup_refresh(self) -> None:
        def job(update, cancel_event) -> str:
            update("checking cached intelligence")
            if cancel_event.is_set():
                return "cancelled before start"
            detail = update_intelligence_job(
                self.config.project_name,
                startup=True,
                run_watchers=False,
                run_orbiters=False,
                run_training=self.config.startup_training,
                run_transfer=False,
                workers=self.config.workers,
                startup_cache_policy=self.config.startup_cache_policy,
                evaluation_mode=self.config.evaluation_mode,
                status_callback=update,
            )
            self.intelligence_state = load_intelligence_state(self.config.project_name)
            self.baseline = load_project_baseline(self.config.project_name)
            self.classifier = self._load_classifier()
            self.timeseries = load_project_timeseries(self.config.project_name)
            self.scenes = load_project_scenes(self.config.project_name)
            if self.intelligence_state is not None:
                self.validation_summary = self._validation_summary_from_state(self.intelligence_state)
                self.evaluation_report = dict(dict(self.intelligence_state.get("models", {})).get("evaluation", {}))
                profile = dict(self.intelligence_state.get("startup_profile", {}))
                slowest = dict(profile.get("slowest_step", {}))
                if slowest:
                    self.messages.append(f"slowest startup step: {slowest.get('name')} {float(slowest.get('elapsed_s', 0.0) or 0.0):.2f}s")
            self.live_message = "background intelligence refresh complete"
            return detail

        self.job_manager.start("startup intelligence refresh", job)

    def _run_council_startup_pipeline(self) -> None:
        progress_rows = initial_progress_rows()
        events: queue.Queue[dict[str, object]] = queue.Queue()
        done = threading.Event()
        skip_steps: set[str] = set()
        state_holder: dict[str, object] = {}
        error_holder: dict[str, object] = {}
        started = time.monotonic()

        def progress(update: dict[str, object]) -> None:
            events.put(dict(update))

        def worker() -> None:
            try:
                checks = run_doctor(self.config.project_name)
                if not doctor_ok(checks):
                    events.put(make_progress("init_project", "running", progress=None, message=f"doctor warnings: {sum(1 for check in checks if check.status != 'ok')}"))
                state_holder["state"] = run_intelligence_update(
                    self.config.project_name,
                    startup=True,
                    run_watchers=self.config.startup_watchers and not self.config.fast_start,
                    run_orbiters=self.config.startup_orbiters and not self.config.fast_start,
                    run_training=self.config.startup_training and not self.config.fast_start,
                    run_transfer=not self.config.fast_start,
                    force_update=self.config.force_startup_update,
                    skip_steps=skip_steps,
                    progress_callback=progress,
                    workers=self.config.workers,
                    startup_cache_policy=self.config.startup_cache_policy,
                    evaluation_mode=self.config.evaluation_mode,
                    profile_startup=self.config.profile_startup,
                )
            except Exception as exc:
                error_holder["error"] = str(exc)
            finally:
                done.set()

        thread = threading.Thread(target=worker, name="dsense-startup-intelligence", daemon=True)
        thread.start()
        self.screen.timeout(150)
        message = "Fast start: loading existing artifacts" if self.config.fast_start else ""
        frame = 0
        while not done.is_set() or not events.empty():
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    break
                name = str(event.get("name", ""))
                if name:
                    progress_rows[name] = event
            current = _current_running_step(progress_rows)
            if current is not None:
                elapsed = float(current.get("elapsed_s", 0.0) or 0.0)
                warning = progress_warning(str(current.get("name", "")), elapsed)
                if warning:
                    current["warning"] = warning
            self._draw_startup_progress(progress_rows, message, frame)
            frame += 1
            key = self.screen.getch()
            if key in (ord("q"), ord("Q")):
                error_holder["error"] = "startup cancelled by user"
                break
            if key in (ord("s"), ord("S")):
                current = _current_running_step(progress_rows)
                current_name = str(current.get("name", "")) if current else ""
                if current_name in OPTIONAL_STARTUP_STEPS:
                    skip_steps.add(current_name)
                    progress_rows[current_name] = make_progress(current_name, "skipped", progress=1.0, message="skip requested")
                    message = f"Skip requested for {current_name}."
                else:
                    message = "This step cannot be skipped safely."
        thread.join(timeout=0.1)
        if error_holder.get("error"):
            self.live_message = str(error_holder["error"])
        state = state_holder.get("state") if isinstance(state_holder.get("state"), dict) else load_intelligence_state(self.config.project_name)
        if state is None:
            state = load_intelligence_state(self.config.project_name) or {}
        self.intelligence_state = state
        self.baseline = load_project_baseline(self.config.project_name)
        self.classifier = self._load_classifier()
        self.timeseries = load_project_timeseries(self.config.project_name)
        self.validation_summary = self._validation_summary_from_state(state) if state else "not run"
        self.evaluation_report = dict(dict(state.get("models", {})).get("evaluation", {})) if state else None
        status = str(state.get("status", "unknown")) if state else "failed"
        council = dict(state.get("council", {})) if state else {}
        detail = f"{status} confidence={council.get('overall_confidence', 0.0)}"
        self.job_manager.add_completed("update intelligence", detail, "done" if status in {"ok", "warning"} else "error", time.monotonic() - started)

        self.scenes = load_project_scenes(self.config.project_name)
        self._draw_startup_progress(progress_rows, "Opening Live View", frame)
        time.sleep(0.4)

    def _run_startup_baseline_step(self, set_step) -> None:
        policy = default_auto_baseline_policy() if self.config.auto_baseline_policy == "auto" else self.config.auto_baseline_policy
        if policy == "off":
            self.config.startup_baseline_status = "Startup baseline: skipped by policy off"
            set_step("startup_baseline", "skipped", "policy off")
            return
        if policy == "missing-only" and not self.config.force_auto_baseline and project_has_usable_baseline(self.config.project_name):
            self.config.startup_baseline_status = "Startup baseline: reused existing model"
            set_step("startup_baseline", "done", "reused existing model")
            return
        try:
            started = time.monotonic()
            scene_id = allocate_scene_id(self.config.project_name)
            duration = max(0.01, float(self.config.auto_baseline_duration))

            def progress(update: dict[str, object]) -> list[dict[str, object]]:
                elapsed_ms = int(update.get("elapsed_ms", 0) or 0)
                duration_ms = int(update.get("duration_ms", int(duration * 1000)) or 1)
                set_step("startup_baseline", "active", f"recording {scene_id}  {elapsed_ms / 1000:0.1f}s / {duration_ms / 1000:0.1f}s")
                return []

            set_step("startup_baseline", "active", f"recording {scene_id}  0.0s / {duration:0.1f}s")
            record_scene(
                project_path(self.config.project_name) / "scenes" / scene_id,
                scene_id,
                "baseline_startup_auto",
                duration,
                max(1, self.config.tick_hz),
                0.0,
                duration,
                0.0,
                "Automatically recorded startup baseline when dSense TUI opened.",
                mode="baseline_auto",
                progress_callback=progress,
                channel_groups=self.config.channel_groups,
            )
            self.config.startup_baseline_status = f"Startup baseline: recorded {scene_id}"
            set_step("startup_baseline", "done", f"recorded {scene_id}")
            self.job_manager.add_completed("startup baseline", f"recorded {scene_id}", "done", time.monotonic() - started)
        except Exception as exc:
            self.config.startup_baseline_status = f"Startup baseline: failed: {exc}"
            set_step("startup_baseline", "warn", f"failed: {exc}")
            self.job_manager.add_completed("startup baseline", f"failed: {exc}", "error")

    def _run_startup_suite_step(self, set_step) -> None:
        if not self.config.startup_suite_enabled:
            set_step("baseline_suite", "skipped", "disabled")
            return
        existing = count_baseline_suite_scenes(self.config.project_name)
        target = max(1, int(self.config.startup_suite_target))
        if existing >= target:
            set_step("baseline_suite", "done", f"{existing} / {target} scenes")
            return
        missing = target - existing

        def progress(update: dict[str, object]) -> None:
            current = existing + int(update.get("current", 0) or 0)
            scene_id = str(update.get("scene_id", "") or "")
            elapsed_ms = int(update.get("elapsed_ms", 0) or 0)
            duration_ms = int(update.get("duration_ms", 0) or 0)
            timing = f"  {elapsed_ms / 1000:0.1f}s / {duration_ms / 1000:0.1f}s" if duration_ms else ""
            detail = f"{current} / {target} scenes"
            if scene_id:
                detail += f"  {scene_id}{timing}"
            set_step("baseline_suite", "active", detail)

        set_step("baseline_suite", "active", f"{existing} / {target} scenes")
        try:
            started = time.monotonic()
            report = run_baseline_suite(
                self.config.project_name,
                target_scenes=missing,
                seed=self.config.startup_suite_seed,
                duration=self.config.startup_suite_duration,
                tick_hz=self.config.tick_hz,
                linux=self.config.startup_suite_linux,
                assume_yes=True,
                label_offset=existing,
                progress_callback=progress,
            )
            recorded = int(report.get("actual_scene_count", 0) or 0)
            set_step("baseline_suite", "done", f"{existing + recorded} / {target} scenes")
            self.job_manager.add_completed("baseline-suite fill", f"{existing + recorded} / {target} scenes", "done", time.monotonic() - started)
        except Exception as exc:
            set_step("baseline_suite", "warn", f"failed: {exc}")
            self.job_manager.add_completed("baseline-suite fill", f"failed: {exc}", "error")

    def _draw_startup_pipeline(self, steps: list[dict[str, object]]) -> None:
        self.screen.erase()
        h, w = self.screen.getmaxyx()
        self._title("LIVE OBSERVATORY", self.config.project_name)
        self._add(2, 2, "Startup intelligence status: local evidence layers are refreshing before live telemetry opens.")
        row = 4
        for step in steps:
            if row >= h - 2:
                break
            status = str(step.get("status", "pending"))
            marker = {"done": "[x]", "active": "[>]", "warn": "[!]", "skipped": "[-]"}.get(status, "[ ]")
            color = self._color(2 if status == "done" else 1 if status == "active" else 3 if status in {"warn", "skipped"} else 0)
            detail = str(step.get("detail", "") or "")
            label = str(step.get("label", ""))
            suffix = f": {detail}" if detail else ""
            self._add(row, 4, clip_text(f"{marker} {label}{suffix}", max(1, w - 8)), color)
            row += 1
        self.screen.refresh()

    def _draw_startup_progress(self, rows: dict[str, dict[str, object]], message: str = "", frame: int = 0) -> None:
        self.screen.erase()
        h, w = self.screen.getmaxyx()
        self._title("dSense Startup Intelligence", self.config.project_name)
        flags = []
        if self.config.fast_start:
            flags.append("fast-start")
        if not self.config.startup_orbiters:
            flags.append("orbiters disabled")
        if not self.config.startup_watchers:
            flags.append("watchers disabled")
        if not self.config.startup_training:
            flags.append("training disabled")
        if self.config.force_startup_update:
            flags.append("force update")
        if flags:
            self._add(1, 2, clip_text(" | ".join(flags), max(1, w - 4)), self._color(3))
        row = 3
        for name in rows:
            if row >= h - 3:
                break
            item = rows[name]
            self._add(row, 2, clip_text(_startup_progress_line(item, frame, max(10, w - 34)), max(1, w - 4)), self._startup_color(str(item.get("status", "pending"))))
            row += 1
        current = _current_running_step(rows)
        current_message = message or (str(current.get("message", "")) if current else "")
        if current_message and h >= 5:
            self._add(h - 3, 2, clip_text(f"Current: {current_message}", max(1, w - 4)), self._color(3 if "cannot" in current_message else 0))
        self._add(h - 2, 2, clip_text("u update again | s skip current safe step | q quit", max(1, w - 4)))
        self.screen.refresh()

    def _startup_color(self, status: str) -> int:
        if status == "done":
            return self._color(2)
        if status == "running":
            return self._color(1)
        if status in {"failed", "skipped"}:
            return self._color(3)
        return 0

    def _configure(self) -> str:
        self.screen.timeout(250)
        fields = [
            ("mode", "Mode"),
            ("scenario", "Preset"),
            ("record_group", "Batch group"),
            ("auto_detect", "Auto events"),
            ("label", "Label"),
            ("duration", "Duration seconds"),
            ("pre_roll", "Pre-roll seconds"),
            ("action", "Action seconds"),
            ("post_roll", "Post-roll seconds"),
            ("repeat", "Repeats"),
            ("tick_hz", "Tick Hz"),
            ("notes", "Notes"),
        ]
        idx = 0
        while True:
            self._update_live_observation()
            self._draw_config(fields, idx)
            key = self.screen.getch()
            if key == -1:
                continue
            current_tab = TABS[self.tab_index]
            if key == 9:
                self._cycle_tab(1)
            elif key in (curses.KEY_BTAB, curses.KEY_LEFT):
                self._cycle_tab(-1)
            elif key == curses.KEY_RIGHT:
                self._cycle_tab(1)
            elif ord("1") <= key <= ord("9"):
                self.tab_index = min(key - ord("1"), len(TABS) - 1)
            elif key == ord("0"):
                self.tab_index = 9 if len(TABS) >= 10 else len(TABS) - 1
            elif current_tab == "Scenes" and key in (curses.KEY_UP, ord("k")):
                self._move_scene_selection(-1)
            elif current_tab == "Scenes" and key in (curses.KEY_DOWN, ord("j")):
                self._move_scene_selection(1)
            elif current_tab == "Capture" and key in (curses.KEY_UP, ord("k")):
                idx = max(0, idx - 1)
            elif current_tab == "Capture" and key in (curses.KEY_DOWN, ord("j")):
                idx = min(len(fields) - 1, idx + 1)
            elif current_tab == "Capture" and key in (10, 13):
                name, title = fields[idx]
                if name == "mode":
                    self._cycle_mode()
                elif name == "scenario":
                    self._cycle_scenario(1)
                elif name == "record_group":
                    self.config.record_group = not self.config.record_group
                elif name == "auto_detect":
                    self.config.auto_detect = not self.config.auto_detect
                else:
                    value = self._prompt(f"{title}", str(getattr(self.config, name)))
                    self._set_config_value(name, value)
            elif key in (ord("m"), ord("M")) and current_tab == "Capture":
                self._cycle_mode()
            elif key in (ord("m"), ord("M")):
                self._mark_live_interval()
            elif key in (ord("p"), ord("P")):
                self._cycle_scenario(1)
            elif key in (ord("o"), ord("O")):
                self._cycle_scenario(-1)
            elif key in (ord("g"), ord("G")):
                self.config.record_group = not self.config.record_group
            elif key in (ord("a"), ord("A")):
                self.config.auto_detect = not self.config.auto_detect
            elif key in (ord("s"), ord("S")) and current_tab == "Capture":
                self.config.duration = self.config.pre_roll + self.config.action + self.config.post_roll
            elif key in (ord("s"), ord("S")):
                self._save_live_snapshot()
            elif key in (ord("r"), ord("R")) and current_tab == "Capture":
                self.channels = scan_channels(groups=self.config.channel_groups)
            elif key in (ord("r"), ord("R")):
                self._prefill_capture_from_live()
                self.tab_index = TABS.index("Capture")
            elif key in (ord("u"), ord("U")):
                self._start_intelligence_job()
            elif key in (ord("t"), ord("T")):
                self._start_training_jobs()
            elif key in (ord("v"), ord("V")):
                self._start_validate_job()
            elif key in (ord("w"), ord("W")):
                self._start_watcher_job()
            elif key in (ord("e"), ord("E")):
                self._start_export_job()
            elif current_tab == "Jobs" and key in (ord("x"), ord("X")):
                self.job_manager.cancel_running()
            elif key in (ord("c"), ord("C")):
                if current_tab == "Capture":
                    return "record"
            elif key in (ord("q"), ord("Q")):
                return "quit"

    def _draw_config(self, fields: list[tuple[str, str]], selected: int) -> None:
        self.screen.erase()
        h, w = self.screen.getmaxyx()
        if h < MIN_TUI_HEIGHT or w < MIN_TUI_WIDTH:
            self._add(0, 0, "Terminal too small. Please enlarge the window.")
            self.screen.refresh()
            return
        self._title("dSense Control Panel", self.config.project_name)
        self._draw_tabs(1, 0, w)
        tab = TABS[self.tab_index]
        content_h = self._content_height(3, h)
        if tab == "Live":
            self._draw_live_tab(3, content_h, w)
        elif tab == "Sense Radar":
            self._draw_sense_radar_tab(3, content_h, w)
        elif tab == "Capture":
            self._draw_record_tab(fields, selected, 3, content_h, w)
        elif tab == "Scenes":
            self._draw_scenes_tab(3, content_h, w)
        elif tab == "Council":
            self._draw_council_tab(3, content_h, w)
        elif tab == "Evaluation":
            self._draw_evaluation_tab(3, content_h, w)
        elif tab == "Watchers":
            self._draw_watcher_tab(3, content_h, w)
        elif tab == "Orbiters":
            self._draw_orbiters_tab(3, content_h, w)
        elif tab == "Transfer":
            self._draw_transfer_tab(3, content_h, w)
        elif tab == "Settings":
            self._draw_settings_tab(3, content_h, w)
        self._draw_program_status(h, w)
        self._add(h - 2, 2, "m mark | r record | u update intelligence | s snapshot | tab tabs | q quit")
        self.screen.refresh()

    def _content_height(self, y: int, screen_h: int) -> int:
        return max(y + 4, self._status_panel_y(screen_h) + 3)

    def _status_panel_y(self, screen_h: int) -> int:
        return max(3, screen_h - self.STATUS_PANEL_HEIGHT - 2)

    def _draw_program_status(self, h: int, w: int) -> None:
        y = self._status_panel_y(h)
        panel_h = min(self.STATUS_PANEL_HEIGHT, max(3, h - y - 2))
        self._box(y, 0, panel_h, w - 1, "Program Status")
        for offset, (line, color) in enumerate(self._program_status_lines(w), start=1):
            if offset >= panel_h - 1:
                break
            self._add(y + offset, 2, clip_text(line, max(1, w - 4)), color)

    def _program_status_lines(self, width: int) -> list[tuple[str, int]]:
        jobs = self.job_manager.snapshot()
        running = [job for job in jobs if job.status == "run"]
        if running:
            job = running[-1]
            detail = job.detail or "working"
            lines = [(f"active: {job.name}  {job.duration_s:0.1f}s  {detail}", self._color(3))]
        else:
            completed = [job for job in jobs if job.status in {"done", "error", "cancelled"}]
            if completed:
                job = completed[-1]
                detail = job.error or job.detail or job.status
                color = self._color(2 if job.status == "done" else 4 if job.status == "error" else 3)
                lines = [(f"idle: last {job.name} {job.status}  {detail}", color)]
            else:
                lines = [("idle: no background job running", self._color(2))]
        if self.live_observation is not None:
            tick = int(self.live_observation.tick)
            elapsed = max(0.001, float(self.live_observation.elapsed_s))
            rate = tick / elapsed
            channel_count = len(dict(self.live_observation.channel_values))
            lines.append((f"live: sampling tick {tick:06d}  {rate:0.1f}Hz  {channel_count} channels  interval {self.live_observation.interval_classification}", self._color(2)))
        else:
            live_text = self.live_message or "waiting for telemetry"
            lines.append((f"live: {live_text}", self._color(3 if self.live_message else 0)))
        state = self.intelligence_state
        if state is None:
            state = load_intelligence_state(self.config.project_name)
            self.intelligence_state = state
        if state:
            council = dict(state.get("council", {}))
            lines.append((f"council: {state.get('status', 'unknown')}  agreement {council.get('agreement', 'unknown')}  confidence {council.get('overall_confidence', 0.0)}", self._color(3 if state.get("status") == "warning" else 2)))
        else:
            lines.append(("council: not updated yet; press u to refresh local intelligence", self._color(3)))
        if self.messages:
            lines.append((f"message: {self.messages[-1]}", self._color(3)))
        return [(clip_text(line, max(1, width - 4)), color) for line, color in lines]

    def _draw_tabs(self, y: int, x: int, width: int) -> None:
        col = x
        for i, tab in enumerate(TABS):
            label = f"[ {tab} ]"
            attr = curses.A_REVERSE | curses.A_BOLD if i == self.tab_index else 0
            if col + len(label) >= x + width - 1:
                break
            self._add(y, col, label, attr)
            col += len(label) + 1

    def _draw_record_tab(self, fields: list[tuple[str, str]], selected: int, y: int, h: int, w: int) -> None:
        left_w = max(30, min(w - 2, w // 2))
        record_h = max(4, min(17, h - y - 4))
        self._box(y, 0, record_h, left_w, "Record")
        self._add(y + 1, 2, f"Project {self.config.project_name}")
        field_y = y + 3
        inner_width = max(1, left_w - 4)
        max_field_rows = max(1, record_h - 4)
        for i, (name, title) in enumerate(fields):
            if field_y >= y + 3 + max_field_rows:
                break
            marker = ">" if i == selected else " "
            value = self._field_value(name)
            attr = curses.A_REVERSE if i == selected else 0
            prefix = f"{marker} {title:<18} "
            value_width = max(1, inner_width - len(prefix))
            if name == "notes":
                wrapped = wrap_text(value, value_width)
                remaining = max(1, y + 3 + max_field_rows - field_y)
                for line_index, wrapped_line in enumerate(wrapped[:remaining]):
                    line_prefix = prefix if line_index == 0 else " " * len(prefix)
                    self._add(field_y, 2, clip_text(line_prefix + wrapped_line, inner_width), attr)
                    field_y += 1
            else:
                self._add(field_y, 2, clip_text(prefix + value, inner_width), attr)
                field_y += 1
        right_x = left_w + 1
        right_w = max(4, w - right_x - 1)
        self._box(y, right_x, 9, right_w, "Status")
        counts = summarize_scene_counts(self.scenes)
        self._add(y + 2, right_x + 2, f"scenes {len(self.scenes)} | baseline {counts['baseline']} | user {counts['user']} | other {counts['other']}")
        baseline_text = "not trained" if self.baseline is None else f"{self.baseline.scene_count} baseline scenes"
        classifier_text = "not trained" if self.classifier is None else f"{self.classifier.scene_count} scenes"
        self._add(y + 4, right_x + 2, f"Baseline: {baseline_text}")
        self._add(y + 5, right_x + 2, f"Classifier: {classifier_text}")
        if self.config.startup_baseline_status:
            self._add(y + 7, right_x + 2, self.config.startup_baseline_status, self._color(2 if "failed" not in self.config.startup_baseline_status.lower() else 3))
        else:
            self._add(y + 7, right_x + 2, "Press c to start capture")

        notes_y = y + 10
        notes_h = max(4, h - notes_y - 4)
        self._box(notes_y, right_x, notes_h, right_w, "Notes")
        self._add_wrapped(notes_y + 2, right_x + 2, self.config.notes or "(no notes)", max(1, right_w - 4), max(1, notes_h - 3))

    def _draw_live_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Live Observatory")
        lines = (
            compact_live_observation_lines(self.live_observation, max(1, w - 4))
            if panel_h < 16 or w < 72
            else live_observation_lines(self.live_observation, max(1, w - 4))
        )
        row = y + 1
        if self.live_message:
            self._add(row, 2, clip_text(self.live_message, max(1, w - 4)), self._color(3))
            row += 1
        if not self.config.startup_intelligence:
            self._add(row, 2, "Startup intelligence disabled for this session; showing existing model state only.", self._color(3))
            row += 1
        for line in lines:
            if row >= y + panel_h - 1:
                break
            color = self._color(3 if "unknown anomaly" in line.lower() or "needs validation" in line.lower() else 1 if line in {"Telemetry", "Sense Radar", "Intelligence Council", "Known Anomalies", "Unknown Anomalies"} else 0)
            self._add(row, 2, clip_text(line, max(1, w - 4)), color)
            row += 1

    def _draw_sense_radar_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Sense Radar - experimental proximity hypothesis")
        proximity = self.live_observation.proximity_hypothesis if self.live_observation is not None else {}
        agreement = self.live_observation.council_agreement if self.live_observation is not None else "unknown"
        lines = [
            "Experimental visualization of local signal anomalies, not literal radar.",
            "It does not prove biological presence; hypotheses need repeated labeled samples.",
            "",
            *sense_radar_lines(proximity, agreement, max(1, w - 4)),
        ]
        for i, line in enumerate(lines[:max(1, panel_h - 2)]):
            self._add(y + 1 + i, 2, clip_text(line, max(1, w - 4)), self._color(3 if "does not prove" in line or "needs validation" in line else 0))

    def _draw_scenes_tab(self, y: int, h: int, w: int) -> None:
        list_w = max(28, min(w - 2, w // 2))
        detail_x = list_w + 1
        detail_w = max(4, w - detail_x - 1)
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, list_w, f"Scenes ({len(self.scenes)})")
        self._box(y, detail_x, panel_h, detail_w, "Scene detail")
        visible_rows = max(1, panel_h - 4)
        self._ensure_scene_visible(visible_rows)
        self._add(y + 1, 2, "Scene ID      Label                 Status Conf")
        rows = self.scenes[self.scene_scroll:self.scene_scroll + visible_rows]
        for i, scene in enumerate(rows):
            actual_index = self.scene_scroll + i
            attr = curses.A_REVERSE if actual_index == self.scene_index else 0
            quality = scene.get("quality", {})
            confidence = quality.get("confidence", "?") if isinstance(quality, dict) else "?"
            accepted = "ok" if scene.get("accepted", False) else "review"
            label = str(scene.get("label", "unknown"))[:20]
            self._add(y + 3 + i, 2, f"{scene.get('scene_id', '?'):<13} {label:<20} {accepted:<6} {confidence}", attr)
        if not self.scenes:
            self._add(y + 3, 2, "No scenes recorded yet.")
            return
        scene = self.scenes[max(0, min(self.scene_index, len(self.scenes) - 1))]
        self._draw_scene_detail(scene, y + 2, detail_x + 2, detail_w - 4, panel_h - 3)

    def _draw_scene_detail(self, scene: dict[str, object], y: int, x: int, width: int, max_lines: int) -> None:
        quality = scene.get("quality", {})
        quality = quality if isinstance(quality, dict) else {}
        lines = scene_detail_lines(scene)
        row = y
        for line in lines[:max(0, max_lines - 5)]:
            self._add(row, x, line)
            row += 1
        notes_lines = max(1, y + max_lines - row)
        self._add(row, x, "Notes:")
        row += 1
        self._add_wrapped(row, x, str(scene.get("notes") or "(no notes)"), width, notes_lines)

    def _draw_channels_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Channels")
        self._add(y + 1, 2, "ID                         Name                         Bit Rate  Status")
        row = y + 3
        for ch in self.channels:
            if row >= y + panel_h - 1:
                break
            status = "online" if ch.get("available") else "offline"
            color = self._color(2 if ch.get("available") else 4)
            line = f"{ch.get('id', ''):<26} {str(ch.get('name', '')):<28} {ch.get('bit', '')!s:<3} {ch.get('rate_hz', '')!s:<5} {status}"
            self._add(row, 2, line, color)
            row += 1
            reason = str(ch.get("reason", ""))
            if reason and reason != "ok" and row < y + panel_h - 1:
                self._add(row, 4, reason[:max(1, w - 8)], self._color(3))
                row += 1

    def _draw_settings_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        left_w = max(28, min(w - 2, w // 2))
        self._box(y, 0, panel_h, left_w, "Settings")
        self._add(y + 1, 2, f"Project: {self.config.project_name}")
        self._add(y + 2, 2, f"Channels: {','.join(self.config.channel_groups)}")
        self._add(y + 3, 2, f"Startup intelligence: {'on' if self.config.startup_intelligence else 'off'}")
        self._add(y + 4, 2, f"Tick Hz: {self.config.tick_hz}")
        self._add(y + 6, 2, "Capture settings live on the Capture tab.")
        right_x = left_w + 1
        right_w = max(4, w - right_x - 1)
        self._box(y, right_x, panel_h, right_w, "Channels")
        self._add(y + 1, right_x + 2, "ID                         Status")
        row = y + 3
        for ch in self.channels[:max(1, panel_h - 5)]:
            status = "online" if ch.get("available") else "offline"
            self._add(row, right_x + 2, clip_text(f"{ch.get('id', ''):<26} {status}", max(1, right_w - 4)), self._color(2 if ch.get("available") else 4))
            row += 1

    def _draw_learn_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Learn")
        if self.baseline is None:
            self._add(y + 2, 2, "Baseline model not trained. Press u to update intelligence.")
            return
        self._add(y + 2, 2, f"Baseline scenes: {self.baseline.scene_count}")
        self._add(y + 3, 2, f"Trained: {self.baseline.trained_utc}")
        self._add(y + 4, 2, f"Threshold: {self.baseline.threshold:g}")
        self._add(y + 6, 2, "Channel                         center          MAD          p95          p99")
        for i, (channel, profile) in enumerate(sorted(self.baseline.channels.items())[:max(1, panel_h - 9)]):
            self._add(y + 8 + i, 2, profile_line(channel, profile))

    def _draw_council_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Intelligence Council")
        if not self.config.startup_intelligence:
            self._add(y + 1, 2, "Startup intelligence disabled for this session.", self._color(3))
        state = self.intelligence_state or load_intelligence_state(self.config.project_name)
        self.intelligence_state = state
        for i, line in enumerate(council_summary_lines(state, max(1, panel_h - 5))[:max(1, panel_h - 3)]):
            if y + 2 + i >= y + panel_h - 1:
                break
            self._add(y + 2 + i, 2, clip_text(line, max(1, w - 4)), self._color(3 if "Warning" in line or line.startswith("  -") else 0))

    def _draw_classify_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Classify")
        for i, line in enumerate(classifier_summary_lines(self.classifier, self.config.auto_detect)[:max(1, panel_h - 3)]):
            color = self._color(2 if self.classifier is not None else 3)
            self._add(y + 2 + i, 2, line, color if i == 0 else 0)
        if self.classifier is not None:
            row = y + 9
            self._add(row, 2, "Label counts:")
            for i, (label, count) in enumerate(sorted(self.classifier.label_counts.items(), key=lambda item: (-item[1], item[0]))[:max(1, h - row - 4)]):
                self._add(row + 2 + i, 4, f"{label}: {count}")

    def _draw_evaluation_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Evaluation")
        report = self.evaluation_report or self._evaluate_project()
        self.evaluation_report = report
        row = y + 2
        self._add(row, 2, "Repeatability", curses.A_BOLD | self._color(1))
        row += 1
        for line in evaluation_repeatability_lines(report):
            if row >= y + panel_h - 2:
                return
            self._add(row, 4, line)
            row += 1
        row += 1
        if row >= y + panel_h - 2:
            return
        self._add(row, 2, "Labels needing more takes", curses.A_BOLD | self._color(1))
        row += 1
        for label, reason in labels_needing_more_takes(report, max(1, (panel_h // 3))):
            if row >= y + panel_h - 2:
                return
            color = self._color(3 if label != "none" else 2)
            self._add(row, 4, clip_text(f"{label:<26} {reason}", max(1, w - 8)), color)
            row += 1
        row += 1
        if row >= y + panel_h - 2:
            return
        self._add(row, 2, "Useful channels", curses.A_BOLD | self._color(1))
        row += 1
        for line in useful_channel_lines(report, max(1, y + panel_h - 2 - row)):
            if row >= y + panel_h - 2:
                return
            self._add(row, 4, clip_text(line, max(1, w - 8)), self._color(2))
            row += 1

    def _draw_jobs_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Jobs")
        self._add(y + 1, 2, "Press u/v/e to start jobs. Press x to request cancellation of the latest running job.")
        self._add(y + 3, 2, "status  job                  duration  detail")
        row = y + 5
        jobs = self.job_manager.snapshot()
        if not jobs:
            self._add(row, 4, "No jobs yet.")
            return
        for job in jobs[-max(1, panel_h - 7):]:
            if row >= y + panel_h - 1:
                break
            color = self._color(2 if job.status == "done" else 3 if job.status in {"run", "cancelled"} else 4 if job.status == "error" else 0)
            status = f"[{job.status[:4]:<4}]"
            detail = job.error or job.detail
            line = f"{status:<7} {job.name:<20} {job.duration_s:>6.2f}s  {detail}"
            self._add(row, 2, clip_text(line, max(1, w - 4)), color)
            row += 1

    def _draw_watcher_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Watcher")
        self._add(y + 2, 2, f"Status: {self.watcher_summary}")
        self._add(y + 3, 2, "Press u to update the full local intelligence stack.")
        self._add(y + 5, 2, "Recent events:")
        events = read_recent_watcher_events(self.config.project_name, max(1, panel_h - 8))
        for i, event in enumerate(events):
            line = f"{event.get('event', '?')} scene={event.get('scene_id', '?')} score={event.get('anomaly_score', '?')} channel={event.get('channel', '?')} pred={event.get('classifier_prediction', '')}"
            self._add(y + 7 + i, 4, line)

    def _draw_orbiters_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Orbiters")
        status = gemma_edge_status()
        self._add(y + 2, 2, f"Gemma Edge: {'on' if status['enabled'] else 'off'} | {status['model']}")
        rows = read_recent_orbiter_summaries(project_path(self.config.project_name), 4)
        if not rows:
            self._add(y + 4, 2, "No orbiter summaries yet.")
            return
        row = y + 4
        for summary in rows:
            if row >= y + panel_h - 2:
                break
            prediction = dict(summary.get("classifier_prediction", {})) if isinstance(summary.get("classifier_prediction"), dict) else {}
            self._add(row, 2, f"{summary.get('scene_id', '?')} anomaly={summary.get('anomaly_score', '?')} pred={prediction.get('label', '?')}")
            row += 1
            row += self._add_wrapped(row, 4, str(summary.get("summary") or "(no summary)"), max(1, w - 8), max(1, y + panel_h - 2 - row))

    def _draw_transfer_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Transfer")
        self._add(y + 2, 2, f"Bundle path: {transfer_bundle_path(self.config.project_name)}")
        self._add(y + 4, 2, f"Last export: {self.transfer_summary}")
        self._add(y + 6, 2, "Press e to export a local transfer bundle.")

    def _draw_validate_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Validate")
        self._add(y + 2, 2, f"Last validation: {self.validation_summary}")
        self._add(y + 3, 2, "Press u for full update or v for validation only.")
        result = self.last_validation_result
        if result is None:
            return
        self._add(y + 5, 2, f"Total {result.total_scenes} | Valid {result.valid_scenes} | Errors {result.error_count} | Warnings {result.warning_count}")
        row = y + 7
        for scene in result.scenes:
            for error in scene.errors:
                if row >= y + panel_h - 2:
                    return
                self._add(row, 4, f"{error.severity} {scene.scene_id} [{error.check}] {error.message}")
                row += 1

    def _draw_help_tab(self, y: int, h: int, w: int) -> None:
        panel_h = max(4, h - y - 4)
        self._box(y, 0, panel_h, w - 1, "Help")
        text = (
            "Tabs: Record configures captures; Scenes inspects recorded scene metadata; Channels shows signal adapters; "
            "Council coordinates local evidence layers; Learn and Classify show trained local models; Evaluation answers repeatability and separation questions; Watcher shows anomaly scans; Orbiters shows local evidence summaries; "
            "Transfer exports local bundles; Validate checks dataset health. Hotkeys: TAB next tab, Shift+TAB or left previous tab, "
            "1-0 jump tabs, c start recording from Record, u update intelligence, v validate, e export, q exit setup. "
            "Scene modes: user scenes need a person during action; baseline scenes mean no intentional event; activity scenes are controlled machine-internal labels. "
            "Pre-roll is control time, action is the labeled window, post-roll is settling time. dSense is local substrate-signal capture and should not be overinterpreted."
        )
        self._add_wrapped(y + 2, 2, text, max(1, w - 4), max(1, panel_h - 3))

    def _cycle_tab(self, direction: int) -> None:
        self.tab_index = tab_index_delta(self.tab_index, direction)

    def _move_scene_selection(self, delta: int) -> None:
        if not self.scenes:
            self.scene_index = 0
            self.scene_scroll = 0
            return
        self.scene_index = max(0, min(len(self.scenes) - 1, self.scene_index + delta))

    def _ensure_scene_visible(self, visible_rows: int) -> None:
        if self.scene_index < self.scene_scroll:
            self.scene_scroll = self.scene_index
        elif self.scene_index >= self.scene_scroll + visible_rows:
            self.scene_scroll = self.scene_index - visible_rows + 1
        self.scene_scroll = max(0, min(self.scene_scroll, max(0, len(self.scenes) - visible_rows)))

    def _draw_phase_dashboard(self, y: int, x: int, h: int, w: int) -> None:
        phases = ["Record", "Learn", "Classify", "Channels", "Watcher", "Orbiters", "Transfer"]
        self._box(y, x, h, w, f"Phase {self.phase_index} {phases[self.phase_index]}")
        lines = self._phase_lines(phases[self.phase_index])
        for i, line in enumerate(lines[:max(1, h - 2)]):
            self._add(y + 2 + i, x + 2, line)

    def _phase_lines(self, phase: str) -> list[str]:
        counts = summarize_scene_counts(self.scenes)
        if phase == "Record":
            latest = self.scenes[-1]["scene_id"] if self.scenes else "none"
            accepted = sum(1 for scene in self.scenes if scene.get("accepted", False))
            return [f"scenes {len(self.scenes)} | accepted {accepted}", f"latest {latest}", f"validation {self.validation_summary}"]
        if phase == "Learn":
            if self.baseline is None:
                return ["baseline not trained", "press t to train"]
            return [f"baseline scenes {self.baseline.scene_count}", f"channels {', '.join(sorted(self.baseline.channels))}", f"threshold {self.baseline.threshold:g}"]
        if phase == "Classify":
            if self.classifier is None:
                return ["classifier not trained", "press t to train"]
            return [f"trained scenes {self.classifier.scene_count}", f"labels {len(self.classifier.label_counts)}", f"baseline/user/other {counts['baseline']}/{counts['user']}/{counts['other']}"]
        if phase == "Channels":
            return [f"{ch['id']} {'online' if ch['available'] else 'offline'} bit {ch['bit']}" for ch in self.channels[:8]]
        if phase == "Watcher":
            lines = [self.watcher_summary, "press w for watcher scan"]
            lines.extend(f"{e.get('event')} {e.get('scene_id')} score={e.get('anomaly_score')}" for e in read_recent_watcher_events(self.config.project_name, 3))
            return lines
        if phase == "Orbiters":
            rows = read_recent_orbiter_summaries(project_path(self.config.project_name), 4)
            status = gemma_edge_status()
            lines = [f"Gemma Edge {'on' if status['enabled'] else 'off'} | {status['model']}"]
            lines.extend(str(row.get("summary", ""))[:60] for row in rows)
            return lines if len(lines) > 1 else lines + ["no orbiter summaries yet"]
        if phase == "Transfer":
            return [self.transfer_summary, f"path {transfer_bundle_path(self.config.project_name)}", "press e to export bundle"]
        return []

    def _field_value(self, name: str) -> str:
        if name == "mode":
            return self.config.mode
        if name == "scenario":
            scenarios = SCENARIO_GROUPS.get(self.config.mode, [])
            if not scenarios:
                return "custom"
            idx = self.config.scenario_index % len(scenarios)
            scenario = scenarios[idx]
            kind = "manual" if scenario.manual else "auto"
            workload = f" workload:{scenario.workload}" if scenario.workload else ""
            return f"{idx + 1}/{len(scenarios)} {scenario.label} ({kind}{workload})"
        if name == "record_group":
            return "yes" if self.config.record_group else "no"
        if name == "auto_detect":
            return "yes" if self.config.auto_detect else "no"
        return str(getattr(self.config, name))

    def _set_config_value(self, name: str, value: str) -> None:
        try:
            if name in {"duration", "pre_roll", "action", "post_roll"}:
                setattr(self.config, name, max(0.0, float(value)))
            elif name in {"repeat", "tick_hz"}:
                setattr(self.config, name, max(1, int(value)))
            else:
                setattr(self.config, name, value)
        except ValueError:
            self.messages.append(f"Ignored invalid value for {name}: {value}")

    def _cycle_mode(self) -> None:
        modes = list(SCENARIO_GROUPS)
        current = modes.index(self.config.mode) if self.config.mode in modes else 0
        self.config.mode = modes[(current + 1) % len(modes)]
        self.config.scenario_index = 0
        self._apply_selected_scenario()

    def _cycle_scenario(self, direction: int) -> None:
        scenarios = SCENARIO_GROUPS.get(self.config.mode, [])
        if not scenarios:
            return
        self.config.scenario_index = (self.config.scenario_index + direction) % len(scenarios)
        self._apply_selected_scenario()

    def _apply_selected_scenario(self) -> None:
        scenarios = SCENARIO_GROUPS.get(self.config.mode, [])
        if not scenarios:
            return
        self._apply_scenario(scenarios[self.config.scenario_index % len(scenarios)])

    def _apply_scenario(self, scenario) -> None:
        self.config.label = scenario.label
        self.config.duration = scenario.duration
        self.config.pre_roll = scenario.pre_roll
        self.config.action = scenario.action_seconds
        self.config.post_roll = scenario.post_roll
        self.config.notes = scenario.notes
        self.config.workload_id = scenario.workload

    def _capture_queue(self) -> list[object | None]:
        if self.config.record_group:
            return list(SCENARIO_GROUPS.get(self.config.mode, [])) or [None]
        return [None] * self.config.repeat

    def _confirm_ready(self) -> None:
        self.screen.timeout(-1)
        self.screen.erase()
        self._title("dSense Interaction Recorder", "Ready")
        self._add(3, 2, f"Project: {self.config.project_name}")
        counts = summarize_scene_counts(self.scenes)
        self._add(4, 2, f"Loaded scenes: {len(self.scenes)} total | baseline {counts['baseline']} | user {counts['user']} | other {counts['other']}")
        baseline_text = "not trained" if self.baseline is None else f"{self.baseline.scene_count} baseline scenes"
        classifier_text = "not trained" if self.classifier is None else f"{self.classifier.scene_count} scenes"
        self._add(5, 2, f"Baseline: {baseline_text} | Classifier: {classifier_text}")
        self._add(6, 2, f"Mode: {self.config.mode}")
        batch_text = "whole preset group" if self.config.record_group else "selected preset/repeats"
        self._add(7, 2, f"Recording: {batch_text}")
        auto_text = "on" if self.config.auto_detect else "off"
        self._add(8, 2, f"Auto events: {auto_text}")
        self._add(9, 2, f"Label: {self.config.label}")
        self._add(10, 2, f"Duration: {self.config.duration:g}s at {self.config.tick_hz} Hz")
        self._add(11, 2, f"Windows: pre {self.config.pre_roll:g}s | action {self.config.action:g}s | post {self.config.post_roll:g}s")
        self._add(13, 2, "During recording: heuristic events are automatic; SPACE/n/q are optional overrides.")
        self._add(15, 2, "Press any key when the scene is physically ready.")
        self.screen.refresh()
        self.screen.getch()

    def _countdown(self) -> None:
        self.screen.nodelay(False)
        for n in range(3, 0, -1):
            self.screen.erase()
            self._title("Prepare Scene", f"Recording starts in {n}")
            self._draw_timeline(5, 2, max(20, self.screen.getmaxyx()[1] - 4), 0)
            self.screen.refresh()
            time.sleep(1)

    def _record_take(self, take: int) -> dict[str, object]:
        init_project(self.config.project_name)
        scene_id = allocate_scene_id(self.config.project_name)
        scene_dir = project_path(self.config.project_name) / "scenes" / scene_id
        self.screen.nodelay(True)
        self._refresh_system_events(0, force_start=True)
        learned_baseline = self.baseline.channels if self.baseline is not None else self.classifier.detector_baseline if self.classifier is not None else None
        threshold = self.baseline.threshold if self.baseline is not None else 6.0
        detector = HeuristicEventDetector(self.config.tick_hz, learned_baseline=learned_baseline, threshold=threshold)
        self.detector_state = detector.state

        def ui_progress(update: dict[str, object]) -> list[dict[str, object]]:
            self.latest = update
            self._update_channel_history(update)
            elapsed_ms = int(update.get("elapsed_ms", 0))
            expected = int(update.get("expected", 1))
            tick = int(update.get("tick", 0))
            if tick + 1 >= expected:
                elapsed_ms = max(elapsed_ms, int(update.get("duration_ms", 0)))
            self._refresh_system_events(elapsed_ms)
            detected = detector.update(update) if self.config.auto_detect else []
            self.detector_state = detector.state
            update["detector"] = {
                "score": self.detector_state.score,
                "channel": self.detector_state.channel,
                "status": self.detector_state.status,
                "threshold": self.detector_state.threshold,
                "samples": self.detector_state.samples,
            }
            marked: list[dict[str, object]] = []
            while True:
                key = self.screen.getch()
                if key == -1:
                    break
                if key == ord(" "):
                    marked.append({"event": "user_interaction_marker", "detail": "manual_space", "t_ms": elapsed_ms})
                elif key in (ord("n"), ord("N")):
                    marked.append({"event": "noise_marker", "detail": "manual_noise", "t_ms": elapsed_ms})
                elif key in (ord("q"), ord("Q")):
                    marked.append({"event": "review_flag", "detail": "manual_review", "t_ms": elapsed_ms})
            self.events.extend(marked)
            self.events.extend(detected)
            for event in [*detected, *marked]:
                shown = dict(event)
                shown.setdefault("source", "user")
                self.live_events.append(shown)
            update["recent_events"] = self.live_events[-12:]
            now = time.monotonic()
            if now - self.last_draw > 0.05:
                self.last_draw = now
                self._draw_recording(take)
            return [*detected, *marked]

        composed_progress = workload_progress_callback(
            self.config.workload_id,
            self.config.pre_roll,
            self.config.action,
            ui_progress,
        )

        def progress(update: dict[str, object]) -> list[dict[str, object]]:
            events = (composed_progress(update) if composed_progress is not None else ui_progress(update)) or []
            for event in events:
                if event.get("source") == "workload":
                    self.live_events.append(dict(event))
            return events

        scene = record_scene(
            scene_dir,
            scene_id,
            self.config.label,
            self.config.duration,
            self.config.tick_hz,
            self.config.pre_roll,
            self.config.action,
            self.config.post_roll,
            self.config.notes,
            progress_callback=progress,
            channel_groups=self.config.channel_groups,
        )
        if self.config.workload_id:
            scene["scenario"] = {
                "mode": self.config.mode,
                "manual": False,
                "automatable": True,
                "workload": self.config.workload_id,
            }
        self.screen.nodelay(False)
        return scene

    def _refresh_system_events(self, elapsed_ms: int, force_start: bool = False) -> None:
        existing = {
            (event.get("source"), event.get("event"))
            for event in self.live_events
        }
        for event in scheduled_scene_events(
            self.config.duration,
            self.config.pre_roll,
            self.config.action,
            self.config.post_roll,
        ):
            key = (event["source"], event["event"])
            if key in existing:
                continue
            if force_start and event["event"] == "scene_start":
                self.live_events.append(event)
                existing.add(key)
            elif int(event["t_ms"]) <= elapsed_ms:
                self.live_events.append(event)
                existing.add(key)

    def _draw_recording(self, take: int) -> None:
        h, w = self.screen.getmaxyx()
        self.screen.erase()
        elapsed = int(self.latest.get("elapsed_ms", 0))
        duration = max(1, int(self.latest.get("duration_ms", int(self.config.duration * 1000))))
        tick = int(self.latest.get("tick", 0)) + 1
        expected = int(self.latest.get("expected", 1))
        phase = str(self.latest.get("phase") or self._phase(elapsed))
        actual_elapsed = max(1, elapsed) / 1000
        rate_hz = tick / actual_elapsed
        self._title("dSense Observatory", f"Take {take}")

        self._box(1, 0, 4, w - 1, "Session")
        self._add(2, 2, clip_text(f"Project: {self.config.project_name:<12} Scene: {self.latest.get('scene_id', '?'):<16} Label: {self.config.label}", max(1, w - 4)))
        self._add(3, 2, clip_text(f"Phase: {phase.upper():<10} {elapsed / 1000:0.1f}s / {duration / 1000:0.1f}s    Frames: {tick}/{expected}    Rate: {rate_hz:0.1f}Hz", max(1, w - 4)))

        self._box(5, 0, 5, w - 1, "Timeline")
        self._draw_event_rail(7, 2, max(10, w - 4), elapsed)

        self._box(10, 0, 4, w - 1, "Signal Watcher")
        self._draw_signal_watcher_summary(12, 2, max(10, w - 4))

        matrix_y = 14
        event_h = min(8, max(4, h // 4))
        matrix_h = max(6, h - matrix_y - event_h - 3)
        self._box(matrix_y, 0, matrix_h, w - 1, "Channel Matrix")
        self._draw_channel_matrix(matrix_y + 2, 2, max(1, w - 4), max(1, matrix_h - 3))

        events_y = matrix_y + matrix_h
        self._box(events_y, 0, max(4, h - events_y - 1), w - 1, "Events")
        self._draw_live_event_list(events_y + 2, 2, max(1, w - 4), max(1, h - events_y - 4))
        self._add(h - 2, 2, "Heuristic events are automatic | SPACE user marker | n noise | q review flag")
        self.screen.refresh()

    def _draw_live_values(self, y: int, x: int, width: int, max_rows: int) -> None:
        values = self.latest.get("values", {})
        values = values if isinstance(values, dict) else {}
        preferred = ["dt_ns", "sleep_drift_ns", "process_ns_estimate"]
        keys = [key for key in preferred if key in values]
        keys.extend(sorted(key for key in values if key not in set(preferred)))
        if not keys:
            self._add(y, x, "No numeric preview values yet.")
            return
        for row, key in enumerate(keys[:max_rows]):
            value = values.get(key, 0)
            self._add(y + row, x, clip_text(f"{key:<26} {format_metric_value(value):>12}", width))

    def _draw_live_channels(self, y: int, x: int, width: int, max_rows: int) -> None:
        channels = self.latest.get("channels", [])
        channels = channels if isinstance(channels, list) else []
        if not channels:
            self._add(y, x, "No channel telemetry yet.")
            return
        header = "ID                    bit  rate  state      value"
        self._add(y, x, clip_text(header, width), curses.A_BOLD)
        for offset, channel in enumerate(channels[:max(0, max_rows - 1)], start=1):
            if not isinstance(channel, dict):
                continue
            state = channel_state_label(channel)
            color = self._color(4 if state == "offline" else 3 if state == "stale" else 2)
            value = channel.get("value")
            line = f"{str(channel.get('id', 'unknown')):<21} {channel.get('bit', '')!s:<4} {float(channel.get('rate_hz', 0) or 0):<5.0f} {state:<10} {format_metric_value(value)}"
            self._add(y + offset, x, clip_text(line, width), color)

    def _update_channel_history(self, update: dict[str, object]) -> None:
        values = update.get("values", {})
        values = values if isinstance(values, dict) else {}
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            history = self.channel_history.setdefault(str(key), deque(maxlen=64))
            history.append(float(value))

    def _baseline_profiles(self) -> dict[str, dict[str, float]]:
        if self.baseline is not None:
            return self.baseline.channels
        if self.classifier is not None:
            return self.classifier.detector_baseline
        return {}

    def _draw_signal_watcher_summary(self, y: int, x: int, width: int) -> None:
        if not self.config.auto_detect:
            self._add(y, x, "Auto detector off")
            return
        state = self.detector_state
        color = self._color(4 if state.status == "event" else 3 if state.status == "watching" else 2)
        line = f"score {state.score:0.1f} / {state.threshold:g}  {state.status.upper():<10} strongest: {state.channel}"
        self._add(y, x, clip_text(line, width), color)

    def _draw_channel_matrix(self, y: int, x: int, width: int, max_rows: int) -> None:
        rows = self._channel_matrix_rows()
        if not rows:
            self._add(y, x, "No channel telemetry yet.")
            return
        header = f"{'channel':<26} {'value':>12} {'MAD/z':>7} {'state':<10} sparkline"
        self._add(y, x, clip_text(header, width), curses.A_BOLD)
        for offset, row in enumerate(rows[:max(0, max_rows - 1)], start=1):
            color = self._color(4 if row["state"] == "offline" else 3 if row["state"] == "stale" else 2)
            line = f"{row['channel']:<26} {row['value']:>12} {row['score']:>7} {row['state']:<10} {row['sparkline']}"
            self._add(y + offset, x, clip_text(line, width), color)

    def _channel_matrix_rows(self) -> list[dict[str, str]]:
        values = self.latest.get("values", {})
        values = values if isinstance(values, dict) else {}
        channels = self.latest.get("channels", [])
        channels = channels if isinstance(channels, list) else []
        profiles = self._baseline_profiles()
        channel_state_by_id = {str(channel.get("id", "")): channel_state_label(channel) for channel in channels if isinstance(channel, dict)}
        rows = []
        for name in sorted(values):
            value = values.get(name)
            history = list(self.channel_history.get(str(name), []))
            score = robust_channel_score(str(name), value, profiles, history)
            state = channel_state_by_id.get(value_channel_id(str(name)), channel_state_by_id.get(str(name), "sampled" if history else "idle"))
            rows.append({
                "channel": str(name),
                "value": format_metric_value(value),
                "score": f"{score:0.1f}",
                "state": state,
                "sparkline": sparkline(history, 10),
            })
        preferred = {"dt_ns": 0, "sleep_drift_ns": 1, "process_ns_estimate": 2}
        return sorted(rows, key=lambda row: (preferred.get(row["channel"], 99), -float(row["score"]), row["channel"]))

    def _draw_live_event_list(self, y: int, x: int, width: int, max_rows: int) -> None:
        visible = self.live_events[-max_rows:]
        for i, event in enumerate(visible):
            source = str(event.get("source", "system"))
            color = self._color(2 if source == "system" else 5 if source == "heuristic" else 3)
            channel = f" {event.get('channel')}" if event.get("channel") else ""
            score = f" score={event.get('score')}" if "score" in event else ""
            line = f"{source:<10} {str(event.get('event', 'event')) + channel:<36} {event.get('t_ms', 0):>6} ms{score}"
            self._add(y + i, x, clip_text(line, width), color)

    def _draw_event_rail(self, y: int, x: int, width: int, elapsed_ms: int) -> None:
        total = max(1, int(self.config.duration * 1000))
        chars = ["."] * width
        for event in scheduled_scene_events(self.config.duration, self.config.pre_roll, self.config.action, self.config.post_roll):
            pos = min(width - 1, int((int(event["t_ms"]) / total) * max(1, width - 1)))
            marker = system_event_marker(str(event["event"]))
            chars[pos] = marker.lower() if int(event["t_ms"]) > elapsed_ms else marker
        for event in self.events:
            pos = min(width - 1, int((int(event.get("t_ms", 0)) / total) * max(1, width - 1)))
            chars[pos] = "!" if event.get("source") == "heuristic" else "*"
        self._add(y, x, "".join(chars))
        self._add(y + 1, x, "S/A/E/X system | ! auto signal | * manual")

    def _draw_detector_meter(self, y: int, x: int, width: int) -> None:
        if not self.config.auto_detect:
            self._add(y, x, "Auto detector off")
            return
        state = self.detector_state
        meter_width = max(8, min(width // 2, 32))
        pct = min(1.0, state.score / max(state.threshold, 1.0))
        fill = int(meter_width * pct)
        color = self._color(4 if state.status == "event" else 3 if state.status == "watching" else 2)
        bar = "[" + "#" * fill + "." * (meter_width - fill) + "]"
        self._add(y, x, f"Signal watcher {bar} {state.score:>4.1f}/{state.threshold:g} {state.status} {state.channel}", color)

    def _review(self, scene: dict[str, object]) -> str:
        self.screen.timeout(-1)
        choices = [("keep", "Keep"), ("retake", "Retake"), ("discard", "Discard")]
        idx = 0
        while True:
            self.screen.erase()
            self._title("Review Take", scene["scene_id"])
            q = scene["quality"]
            self._add(3, 2, f"Confidence: {q['confidence']} | Frames: {q['actual_frames']}/{q['expected_frames']} | Markers: {scene.get('user_event_count', 0)}")
            self._add(5, 2, f"Checksum ok: {q['checksum_ok']} | Dropped/late estimate: {q['dropped_or_late_estimate']}")
            self._add(7, 2, f"Stored in: {project_path(self.config.project_name) / 'scenes' / scene['scene_id']}")
            for i, (_, label) in enumerate(choices):
                self._add(10, 2 + i * 14, label, curses.A_REVERSE if i == idx else 0)
            self._add(13, 2, "Use left/right, then Enter.")
            self.screen.refresh()
            key = self.screen.getch()
            if key in (curses.KEY_LEFT, ord("h")):
                idx = max(0, idx - 1)
            elif key in (curses.KEY_RIGHT, ord("l"), 9):
                idx = min(len(choices) - 1, idx + 1)
            elif key in (10, 13):
                return choices[idx][0]

    def _complete(self, results: list[dict[str, object]]) -> None:
        self.screen.timeout(-1)
        kept = sum(1 for scene in results if scene.get("accepted"))
        self.screen.erase()
        self.scenes = load_project_scenes(self.config.project_name)
        try:
            self.intelligence_state = run_intelligence_update(
                self.config.project_name,
                startup=False,
                run_watchers=False,
                run_orbiters=False,
                run_training=True,
                run_transfer=False,
            )
            self.baseline = load_project_baseline(self.config.project_name)
            self.classifier = self._load_classifier()
            self.timeseries = load_project_timeseries(self.config.project_name)
            self.evaluation_report = dict(dict(self.intelligence_state.get("models", {})).get("evaluation", {}))
        except Exception:
            self.baseline = self._train_baseline()
            self.classifier = self._train_classifier()
            self.evaluation_report = self._evaluate_project()
        self._title("Capture Complete", f"{kept}/{len(results)} accepted | {len(self.scenes)} total in {self.config.project_name}")
        for i, scene in enumerate(results[: self.screen.getmaxyx()[0] - 6]):
            status = "kept" if scene.get("accepted") else "not accepted"
            self._add(4 + i, 2, f"{scene['scene_id']} {scene['label']} confidence={scene['quality']['confidence']} {status}")
        self._add(self.screen.getmaxyx()[0] - 2, 2, "Press any key to return to the opening screen.")
        self.screen.refresh()
        self.screen.getch()

    def _validate_project_summary(self) -> str:
        result, summary = validate_project(self.config.project_name)
        self.last_validation_result = result
        return summary

    def _evaluate_project(self) -> dict[str, object] | None:
        try:
            return evaluate_project_job(self.config.project_name)
        except (OSError, ValueError):
            return None

    def _update_live_observation(self) -> None:
        try:
            if self.live_sampler is None:
                self.live_sampler = LiveSampler(self.config.channel_groups, min(max(1, self.config.tick_hz), 20))
                self.live_started_monotonic = time.monotonic()
            values, status, _ = self.live_sampler.sample()
            self.live_rows.append(values)
            tick = self.live_sampler.tick
            elapsed = max(0.001, time.monotonic() - self.live_started_monotonic)
            watcher_events = read_recent_watcher_events(self.config.project_name, 5)
            self.live_observation = build_live_observation(
                self.config.project_name,
                tick=tick,
                elapsed_s=elapsed,
                channel_values=values,
                channel_status=status,
                recent_rows=list(self.live_rows),
                baseline=self.baseline,
                classifier=self.classifier,
                timeseries=self.timeseries,
                council_state=self.intelligence_state,
                watcher_events=watcher_events,
            )
            self.live_writer.maybe_write(self.live_observation)
        except Exception as exc:
            self.live_message = f"Live telemetry unavailable: {exc}"
            self._close_live_sampler()

    def _close_live_sampler(self) -> None:
        if self.live_sampler is None:
            return
        self.live_sampler.close()
        self.live_sampler = None

    def _mark_live_interval(self) -> None:
        if self.live_observation is None:
            self.live_message = "No live interval available to mark yet."
            return
        self.live_writer.maybe_write(self.live_observation, force=True, event="user_mark_interval")
        self.live_message = "Marked current live interval."

    def _save_live_snapshot(self) -> None:
        if self.live_observation is None:
            self.live_message = "No live snapshot available yet."
            return
        path = save_live_snapshot(self.config.project_name, self.live_observation)
        self.live_writer.maybe_write(self.live_observation, force=True, event="snapshot_saved")
        self.live_message = f"Saved live snapshot: {path}"

    def _prefill_capture_from_live(self) -> None:
        if self.live_observation is None:
            self.config.label = "live_disturbance_review"
            self.config.notes = "Opened Capture from Live Observatory before a live observation was available."
            return
        unknown = self.live_observation.unknown_anomalies[:1]
        known = self.live_observation.known_anomalies[:1]
        top = dict((unknown or known or [{}])[0])
        recommended = str(top.get("action", "record scene"))
        label_hint = str(top.get("name", "live_disturbance_review")).replace("?", "")
        if self.live_observation.interval_classification == "normal":
            label_hint = "live_control_interval"
            recommended = "ignore once or mark interval"
        self.config.label = label_hint if label_hint and label_hint != "unclassified pattern" else "live_unknown_disturbance"
        self.config.notes = (
            "Opened from Live Observatory. "
            f"interval={self.live_observation.interval_classification}; "
            f"baseline_score={self.live_observation.baseline_score}; "
            f"watcher_score={self.live_observation.watcher_score}; "
            f"agreement={self.live_observation.council_agreement}; "
            f"recommended_action={recommended}. "
            "Experimental local signal hypothesis; needs repeated validation."
        )
        self.live_message = "Capture prefilled from current live anomaly context. Press c from Capture to record."

    def _validation_summary_from_state(self, state: dict[str, object]) -> str:
        for step in list(state.get("steps", [])):
            row = dict(step)
            if row.get("name") == "validate":
                summary = dict(row.get("summary", {}))
                return f"{summary.get('valid_scenes', 0)}/{summary.get('total_scenes', 0)} valid, {summary.get('errors', 0)} errors, {summary.get('warnings', 0)} warnings"
        return "not run"

    def _start_intelligence_job(self) -> None:
        def job(update, cancel_event) -> str:
            update("updating local evidence layers")
            if cancel_event.is_set():
                return "cancelled before start"
            detail = update_intelligence_job(
                self.config.project_name,
                run_watchers=self.config.startup_watchers,
                run_orbiters=self.config.startup_orbiters,
                run_training=True,
                run_transfer=True,
                workers=self.config.workers,
                startup_cache_policy=self.config.startup_cache_policy,
                evaluation_mode="full",
                status_callback=update,
            )
            self.intelligence_state = load_intelligence_state(self.config.project_name)
            self.baseline = load_project_baseline(self.config.project_name)
            self.classifier = self._load_classifier()
            self.timeseries = load_project_timeseries(self.config.project_name)
            self.scenes = load_project_scenes(self.config.project_name)
            if self.intelligence_state is not None:
                self.validation_summary = self._validation_summary_from_state(self.intelligence_state)
                self.evaluation_report = dict(dict(self.intelligence_state.get("models", {})).get("evaluation", {}))
            return detail

        self.job_manager.start("update intelligence", job)

    def _start_training_jobs(self) -> None:
        def baseline_job(update, cancel_event) -> str:
            update("training")
            if cancel_event.is_set():
                return "cancelled before start"
            self.baseline = self._train_baseline()
            return "not enough accepted baseline scenes" if self.baseline is None else f"{self.baseline.scene_count} scenes"

        def classifier_job(update, cancel_event) -> str:
            update("training")
            if cancel_event.is_set():
                return "cancelled before start"
            self.classifier = self._train_classifier()
            self.evaluation_report = self._evaluate_project()
            return "not enough accepted scenes" if self.classifier is None else f"{self.classifier.scene_count} scenes"

        self.job_manager.start("train baseline", baseline_job)
        self.job_manager.start("train classifier", classifier_job)

    def _start_validate_job(self) -> None:
        def job(update, cancel_event) -> str:
            update("checking scenes")
            if cancel_event.is_set():
                return "cancelled before start"
            self.validation_summary = self._validate_project_summary()
            self.evaluation_report = self._evaluate_project()
            return self.validation_summary

        self.job_manager.start("validate dataset", job)

    def _start_watcher_job(self) -> None:
        def job(update, cancel_event) -> str:
            update("running scan")
            if cancel_event.is_set():
                return "cancelled before start"
            self.watcher_summary = self._run_watcher_summary()
            self.scenes = load_project_scenes(self.config.project_name)
            self.baseline = self._train_baseline()
            self.classifier = self._train_classifier()
            self.evaluation_report = self._evaluate_project()
            return self.watcher_summary

        self.job_manager.start("watcher scan", job)

    def _start_export_job(self) -> None:
        def job(update, cancel_event) -> str:
            update("writing bundle")
            if cancel_event.is_set():
                return "cancelled before start"
            self.transfer_summary = self._export_transfer_summary()
            return self.transfer_summary

        self.job_manager.start("export transfer", job)

    def _run_watcher_summary(self) -> str:
        return run_watcher_job(self.config.project_name, self.config.channel_groups, duration=3.0, tick_hz=50)

    def _export_transfer_summary(self) -> str:
        return export_transfer_job(self.config.project_name)

    def _prompt(self, title: str, default: str) -> str:
        h, w = self.screen.getmaxyx()
        curses.echo()
        curses.curs_set(1)
        self._add(h - 2, 2, " " * (w - 4))
        self._add(h - 2, 2, f"{title} [{default}]: ")
        self.screen.refresh()
        input_x = min(max(2, len(title) + len(default) + 7), max(2, w - 2))
        raw = self.screen.getstr(h - 2, input_x, max(1, w - input_x - 1))
        curses.noecho()
        curses.curs_set(0)
        value = raw.decode("utf-8").strip()
        return value or default

    def _title(self, title: str, subtitle: str = "") -> None:
        self._add(0, 2, title, curses.A_BOLD | self._color(1))
        if subtitle:
            self._add(0, max(4, self.screen.getmaxyx()[1] - len(subtitle) - 2), subtitle)

    def _box(self, y: int, x: int, h: int, w: int, title: str) -> None:
        screen_h, screen_w = self.screen.getmaxyx()
        if h < 3 or w < 4 or y < 0 or y >= screen_h or x < 0 or x >= screen_w:
            return
        max_w = max(4, min(w, screen_w - x))
        try:
            self.screen.addstr(y, x, "+" + "-" * (max_w - 2) + "+")
            for row in range(y + 1, min(y + h - 1, screen_h - 1)):
                self.screen.addstr(row, x, "|")
                self.screen.addstr(row, min(x + max_w - 1, screen_w - 1), "|")
            self.screen.addstr(min(y + h - 1, screen_h - 1), x, "+" + "-" * (max_w - 2) + "+")
        except curses.error:
            return
        self._add(y, x + 2, f" {title} ", curses.A_BOLD)

    def _draw_bar(self, y: int, x: int, width: int, pct: float) -> None:
        fill = max(0, min(width, int(width * pct)))
        self._add(y, x, "[" + "#" * fill + "." * (width - fill) + "]")

    def _metric(self, y: int, x: int, label: str, value: object, scale: int) -> None:
        number = abs(int(value or 0))
        pct = min(1.0, number / max(1, scale))
        self._add(y, x, f"{label:<20} {int(value or 0):>12}")
        self._draw_bar(y + 1, x, 28, pct)

    def _draw_timeline(self, y: int, x: int, width: int, elapsed_ms: int) -> None:
        total = max(1, int(self.config.duration * 1000))
        start = int(self.config.pre_roll * 1000)
        end = int((self.config.pre_roll + self.config.action) * 1000)
        chars = []
        for i in range(width):
            t = int(total * i / max(1, width - 1))
            if t < start:
                chars.append("p")
            elif t <= end:
                chars.append("A")
            else:
                chars.append("o")
        cursor = min(width - 1, int((elapsed_ms / total) * max(1, width - 1)))
        chars[cursor] = "|"
        self._add(y, x, "".join(chars))
        self._add(y + 1, x, "p=pre-roll A=action o=post-roll")

    def _phase(self, elapsed_ms: int) -> str:
        start = int(self.config.pre_roll * 1000)
        end = int((self.config.pre_roll + self.config.action) * 1000)
        if elapsed_ms < start:
            return "pre-roll"
        if elapsed_ms <= end:
            return "action"
        return "post-roll"

    def _add(self, y: int, x: int, text: str, attr: int = 0) -> None:
        h, w = self.screen.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        safe = text[: max(0, w - x - 1)]
        try:
            self.screen.addstr(y, x, safe, attr)
        except curses.error:
            pass

    def _add_wrapped(self, y: int, x: int, text: str, width: int, max_lines: int, attr: int = 0) -> int:
        lines = wrap_text(text, width)[:max(0, max_lines)]
        for i, line in enumerate(lines):
            self._add(y + i, x, line, attr)
        return len(lines)

    def _color(self, pair: int) -> int:
        return curses.color_pair(pair) if curses.has_colors() else 0


def run_curses_tui(config: CaptureConfig) -> list[dict[str, object]]:
    return curses.wrapper(lambda screen: SceneRecorderTUI(screen, config).run())


def run_tui(config: CaptureConfig) -> list[dict[str, object]]:
    return run_curses_tui(config)


def load_project_scenes(project_name: str) -> list[dict[str, object]]:
    scenes_root = project_path(project_name) / "scenes"
    if not scenes_root.exists():
        return []
    scenes = []
    for path in sorted(scenes_root.glob("scene_*/scene.json")):
        try:
            scenes.append(read_json(path))
        except (OSError, ValueError):
            scenes.append({"scene_id": path.parent.name, "label": "unreadable", "accepted": False})
    return scenes


def _step_detail(step: dict[str, object]) -> str:
    summary = step.get("summary", {})
    if isinstance(summary, dict):
        if summary.get("error"):
            return str(summary.get("error"))
        for key in ("path", "scene_count", "valid_scenes", "event_count", "summary_count", "total_scenes"):
            if key in summary:
                return f"{key}={summary.get(key)}"
    return ""


def _ui_status(status: str) -> str:
    return {
        "running": "active",
        "ok": "done",
        "warning": "warn",
        "failed": "warn",
        "skipped": "skipped",
    }.get(status, status)


def _current_running_step(rows: dict[str, dict[str, object]]) -> dict[str, object] | None:
    for row in rows.values():
        if row.get("status") == "running":
            return row
    return None


def _startup_progress_line(item: dict[str, object], frame: int, bar_width: int = 20) -> str:
    status = str(item.get("status", "pending"))
    label = str(item.get("label", item.get("name", "")))
    progress = item.get("progress")
    elapsed = float(item.get("elapsed_s", 0.0) or 0.0)
    message = str(item.get("error") or item.get("warning") or item.get("message") or "")
    if isinstance(progress, (int, float)):
        pct = max(0.0, min(1.0, float(progress)))
        filled = int(round(pct * bar_width))
        bar = "█" * filled + "░" * max(0, bar_width - filled)
        pct_text = f"{pct * 100:3.0f}%"
    elif status == "running":
        spinner = "◐◓◑◒"[frame % 4]
        bar = f"{spinner} working...".ljust(bar_width)
        pct_text = "   "
    else:
        bar = "░" * bar_width
        pct_text = "  0%"
    count = ""
    current = item.get("current")
    total = item.get("total")
    if current is not None and total is not None:
        count = f" {current}/{total}"
    return f"[{status:<7}] {label:<16} {bar} {pct_text} {elapsed:5.1f}s{count}  {message}".rstrip()
