from __future__ import annotations

import curses
import time
from dataclasses import dataclass

from .autotest import validate_dataset
from .baseline import BaselineModel, train_and_save_project_baseline
from .classifier import SceneClassifierModel, train_and_save_project_classifier
from .event_detector import DetectorState, HeuristicEventDetector
from .gemma_edge import gemma_edge_status
from .manifest import allocate_scene_id, init_project, project_path, scan_channels
from .orbiters import read_recent_orbiter_summaries
from .recorder import record_scene
from .scenarios import SCENARIO_GROUPS
from .transfer import export_transfer_bundle, transfer_bundle_path
from .utils.files import read_json, write_json
from .watcher import read_recent_watcher_events, run_watcher_scan


@dataclass
class CaptureConfig:
    project_name: str
    mode: str = "user"
    scenario_index: int = 0
    record_group: bool = False
    auto_detect: bool = True
    label: str = "user_interaction"
    duration: float = 10.0
    pre_roll: float = 2.0
    action: float = 5.0
    post_roll: float = 3.0
    repeat: int = 1
    tick_hz: int = 100
    notes: str = ""


class SceneRecorderTUI:
    def __init__(self, screen, config: CaptureConfig):
        self.screen = screen
        self.config = config
        self.channels = scan_channels()
        self.scenes = load_project_scenes(config.project_name)
        self.baseline = self._train_baseline()
        self.classifier = self._train_classifier()
        self.phase_index = 0
        self.validation_summary = "not run"
        self.watcher_summary = "idle"
        self.transfer_summary = "not exported"
        self.messages: list[str] = []
        self.events: list[dict[str, object]] = []
        self.live_events: list[dict[str, object]] = []
        self.latest: dict[str, object] = {}
        self.detector_state = DetectorState()
        self.last_draw = 0.0
        self._apply_selected_scenario()

    def _train_classifier(self) -> SceneClassifierModel | None:
        try:
            model = train_and_save_project_classifier(self.config.project_name)
        except (OSError, ValueError):
            return None
        return model if model.scene_count else None

    def _train_baseline(self) -> BaselineModel | None:
        try:
            model = train_and_save_project_baseline(self.config.project_name)
        except (OSError, ValueError):
            return None
        return model if model.scene_count else None

    def run(self) -> list[dict[str, object]]:
        curses.curs_set(0)
        self.screen.nodelay(False)
        self.screen.keypad(True)
        self._setup_colors()
        all_results = []
        while True:
            if not self._configure():
                return all_results
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
            self.scenes = load_project_scenes(self.config.project_name)
            if decision == "retake":
                queue.insert(take, scenario)
            take += 1
        return results

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

    def _configure(self) -> bool:
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
            self._draw_config(fields, idx)
            key = self.screen.getch()
            if key in (curses.KEY_UP, ord("k")):
                idx = max(0, idx - 1)
            elif key in (curses.KEY_DOWN, ord("j"), 9):
                idx = min(len(fields) - 1, idx + 1)
            elif key in (10, 13):
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
            elif key in (ord("m"), ord("M")):
                self._cycle_mode()
            elif key in (ord("p"), ord("P"), curses.KEY_RIGHT):
                self._cycle_scenario(1)
            elif key in (ord("o"), ord("O"), curses.KEY_LEFT):
                self._cycle_scenario(-1)
            elif key in (ord("g"), ord("G")):
                self.config.record_group = not self.config.record_group
            elif key in (ord("a"), ord("A")):
                self.config.auto_detect = not self.config.auto_detect
            elif key in (ord("s"), ord("S")):
                self.config.duration = self.config.pre_roll + self.config.action + self.config.post_roll
            elif key in (ord("r"), ord("R")):
                self.channels = scan_channels()
            elif key in (ord("t"), ord("T")):
                self.baseline = self._train_baseline()
                self.classifier = self._train_classifier()
            elif key in (ord("v"), ord("V")):
                self.validation_summary = self._validate_project_summary()
            elif key in (ord("w"), ord("W")):
                self.watcher_summary = self._run_watcher_summary()
                self.scenes = load_project_scenes(self.config.project_name)
                self.baseline = self._train_baseline()
                self.classifier = self._train_classifier()
            elif key in (ord("e"), ord("E")):
                self.transfer_summary = self._export_transfer_summary()
            elif ord("1") <= key <= ord("7"):
                self.phase_index = key - ord("1")
            elif key in (ord("c"), ord("C")):
                return True
            elif key in (ord("q"), ord("Q")):
                return False

    def _draw_config(self, fields: list[tuple[str, str]], selected: int) -> None:
        self.screen.erase()
        h, w = self.screen.getmaxyx()
        self._title("dSense Interaction Recorder", "Configure capture")
        self._box(2, 0, min(17, h - 4), max(30, min(w - 1, w // 2)), "Scene")
        for i, (name, title) in enumerate(fields):
            marker = ">" if i == selected else " "
            value = self._field_value(name)
            self._add(4 + i, 2, f"{marker} {title:<20} {value}", curses.A_REVERSE if i == selected else 0)

        self._draw_phase_dashboard(20, 0, max(4, h - 24), max(30, min(w - 1, w // 2)))

        right_x = max(32, w // 2 + 1)
        self._box(2, right_x, min(9, h - 4), max(28, w - right_x - 1), "Channels")
        for i, ch in enumerate(self.channels[:5]):
            status = "online" if ch["available"] else "offline"
            color = self._color(2 if ch["available"] else 4)
            self._add(4 + i, right_x + 2, f"{ch['id']:<18} {status:<8} bit {ch['bit']}", color)

        classifier_y = 11
        classifier_h = 8
        self._box(classifier_y, right_x, classifier_h, max(28, w - right_x - 1), "Classifier")
        for i, line in enumerate(classifier_summary_lines(self.classifier, self.config.auto_detect)[:classifier_h - 2]):
            color = self._color(2 if self.classifier is not None else 3)
            self._add(classifier_y + 2 + i, right_x + 2, line, color if i == 0 else 0)

        scene_y = 20
        counts = summarize_scene_counts(self.scenes)
        self._box(scene_y, right_x, max(4, h - scene_y - 4), max(28, w - right_x - 1), f"Project scenes ({len(self.scenes)})")
        self._add(scene_y + 1, right_x + 2, f"baseline {counts['baseline']} | user {counts['user']} | other {counts['other']}")
        visible_scenes = self.scenes[-max(1, h - scene_y - 7):]
        if visible_scenes:
            for i, scene in enumerate(visible_scenes):
                label = str(scene.get("label", "unknown"))[:18]
                accepted = "ok" if scene.get("accepted", False) else "review"
                confidence = scene.get("quality", {}).get("confidence", "?") if isinstance(scene.get("quality"), dict) else "?"
                self._add(scene_y + 3 + i, right_x + 2, f"{scene.get('scene_id', '?'):<13} {label:<18} {accepted:<6} {confidence}")
        else:
            self._add(scene_y + 3, right_x + 2, "No scenes yet. New captures will land here.")

        self._add(h - 3, 2, "1-7 phases | t train | v validate | w watcher | e export | c starts | q exits")
        self.screen.refresh()

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
            return f"{idx + 1}/{len(scenarios)} {scenarios[idx].label}"
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

    def _capture_queue(self) -> list[object | None]:
        if self.config.record_group:
            return list(SCENARIO_GROUPS.get(self.config.mode, [])) or [None]
        return [None] * self.config.repeat

    def _confirm_ready(self) -> None:
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

        def progress(update: dict[str, object]) -> list[dict[str, object]]:
            self.latest = update
            elapsed_ms = int(update.get("elapsed_ms", 0))
            expected = int(update.get("expected", 1))
            tick = int(update.get("tick", 0))
            if tick + 1 >= expected:
                elapsed_ms = max(elapsed_ms, int(update.get("duration_ms", 0)))
            self._refresh_system_events(elapsed_ms)
            detected = detector.update(update) if self.config.auto_detect else []
            self.detector_state = detector.state
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
            now = time.monotonic()
            if now - self.last_draw > 0.05:
                self.last_draw = now
                self._draw_recording(take)
            return [*detected, *marked]

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
        )
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
        pct = min(1.0, elapsed / duration)
        tick = int(self.latest.get("tick", 0)) + 1
        expected = int(self.latest.get("expected", 1))

        total_events = len(self.live_events)
        manual_events = sum(1 for event in self.events if event.get("source", "user") == "user" or event.get("event") in {"user_interaction_marker", "noise_marker", "review_flag"})
        auto_events = sum(1 for event in self.events if event.get("source") == "heuristic")
        self._title("Recording Interaction", f"Take {take}")
        self._box(2, 0, 7, w - 1, "Live overview")
        self._add(4, 2, f"{self.config.label}  {elapsed / 1000:5.1f}s / {duration / 1000:5.1f}s")
        self._draw_bar(5, 2, max(10, w - 4), pct)
        self._add(6, 2, f"Frames {tick}/{expected} | Events {total_events} ({auto_events} auto, {manual_events} manual) | Availability mask {self.latest.get('availability_mask', 0)}")

        left_w = max(30, w // 2)
        self._box(10, 0, 9, left_w, "Timing")
        self._metric(12, 2, "dt ns", self.latest.get("dt_ns", 0), 20_000_000)
        self._metric(14, 2, "sleep drift ns", self.latest.get("sleep_drift_ns", 0), 10_000_000)
        self._metric(16, 2, "process ns estimate", self.latest.get("process_ns_estimate", 0), 10_000_000)

        self._box(10, left_w + 1, 9, max(20, w - left_w - 1), "Phase")
        self._draw_timeline(12, left_w + 3, max(10, w - left_w - 5), elapsed)
        phase = self._phase(elapsed)
        self._add(15, left_w + 3, f"Current phase: {phase}", self._color(1))
        self._draw_event_rail(17, left_w + 3, max(10, w - left_w - 5), elapsed)
        self._draw_detector_meter(19, left_w + 3, max(10, w - left_w - 5))

        self._box(22, 0, max(4, h - 24), w - 1, "Events")
        visible = self.live_events[-max(1, h - 27):]
        for i, event in enumerate(visible):
            source = str(event.get("source", "system"))
            color = self._color(2 if source == "system" else 5 if source == "heuristic" else 3)
            score = f" score={event.get('score')}" if "score" in event else ""
            self._add(24 + i, 2, f"{source:<9} {event.get('event', 'event'):<25} {event.get('t_ms', 0):>6} ms{score}", color)
        self._add(h - 2, 2, "Heuristic events are automatic | SPACE user marker | n noise | q review flag")
        self.screen.refresh()

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
        kept = sum(1 for scene in results if scene.get("accepted"))
        self.screen.erase()
        self.scenes = load_project_scenes(self.config.project_name)
        self.baseline = self._train_baseline()
        self.classifier = self._train_classifier()
        self._title("Capture Complete", f"{kept}/{len(results)} accepted | {len(self.scenes)} total in {self.config.project_name}")
        for i, scene in enumerate(results[: self.screen.getmaxyx()[0] - 6]):
            status = "kept" if scene.get("accepted") else "not accepted"
            self._add(4 + i, 2, f"{scene['scene_id']} {scene['label']} confidence={scene['quality']['confidence']} {status}")
        self._add(self.screen.getmaxyx()[0] - 2, 2, "Press any key to return to the opening screen.")
        self.screen.refresh()
        self.screen.getch()

    def _validate_project_summary(self) -> str:
        result = validate_dataset(self.config.project_name)
        return f"{result.valid_scenes}/{result.total_scenes} valid, {result.error_count} errors, {result.warning_count} warnings"

    def _run_watcher_summary(self) -> str:
        result = run_watcher_scan(self.config.project_name, duration=3.0, tick_hz=50)
        detected = len(result.get("detected", []))
        scene = dict(result.get("scene", {}))
        return f"watcher saved {scene.get('scene_id', '?')} with {detected} auto events"

    def _export_transfer_summary(self) -> str:
        bundle = export_transfer_bundle(self.config.project_name)
        return f"exported {bundle.get('total_scenes', 0)} scenes"

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

    def _color(self, pair: int) -> int:
        return curses.color_pair(pair) if curses.has_colors() else 0


def run_tui(config: CaptureConfig) -> list[dict[str, object]]:
    return curses.wrapper(lambda screen: SceneRecorderTUI(screen, config).run())


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


def summarize_scene_counts(scenes: list[dict[str, object]]) -> dict[str, int]:
    counts = {"baseline": 0, "user": 0, "other": 0}
    for scene in scenes:
        label = str(scene.get("label", ""))
        if label.startswith("baseline_"):
            counts["baseline"] += 1
        elif label.startswith("user_") or label.startswith("person_") or label in {"Approach", "typing_burst", "mouse_activity", "door_open_close", "phone_near_computer"}:
            counts["user"] += 1
        else:
            counts["other"] += 1
    return counts


def classifier_summary_lines(model: SceneClassifierModel | None, auto_detect: bool) -> list[str]:
    if model is None:
        return [
            "No classifier trained yet",
            "Accepted scenes will train it",
            "Press t to retry training",
        ]

    label_count = len(model.label_counts)
    channels = ", ".join(sorted(model.detector_baseline)) or "none"
    auto_text = "using learned baseline" if auto_detect and model.detector_baseline else "not used by auto events"
    top_labels = sorted(model.label_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    label_text = ", ".join(f"{label}:{count}" for label, count in top_labels) or "none"
    trained = model.trained_utc.replace("T", " ").replace("Z", " UTC")
    if "." in trained:
        trained = trained.split(".", 1)[0] + " UTC"
    return [
        "Active",
        f"trained scenes {model.scene_count} | baseline {model.baseline_scene_count}",
        f"labels {label_count} | {auto_text}",
        f"channels {channels}",
        f"top {label_text}",
        f"trained {trained}",
    ]


def scheduled_scene_events(duration: float, pre_roll: float, action: float, post_roll: float) -> list[dict[str, object]]:
    action_start_ms = int(pre_roll * 1000)
    action_end_ms = int((pre_roll + action) * 1000)
    return [
        {"t_ms": 0, "event": "scene_start", "source": "system"},
        {"t_ms": action_start_ms, "event": "action_start", "source": "system"},
        {"t_ms": action_end_ms, "event": "action_end", "source": "system"},
        {"t_ms": int(duration * 1000), "event": "scene_end", "source": "system"},
    ]


def system_event_marker(event_name: str) -> str:
    return {
        "scene_start": "S",
        "action_start": "A",
        "action_end": "E",
        "scene_end": "X",
    }.get(event_name, "?")
