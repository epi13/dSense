from __future__ import annotations

import os

from .manifest import scan_channels
from .council import load_intelligence_state
from .tui_jobs import evaluate_project_job, train_models
from .tui_render import council_summary_lines, evaluation_repeatability_lines, labels_needing_more_takes, summarize_scene_counts, useful_channel_lines
from .tui_state import AppState, CaptureConfig


def run_tui(config: CaptureConfig) -> list[dict[str, object]]:
    backend = os.environ.get("DSENSE_TUI_BACKEND", "auto").strip().lower()
    startup_visible_in_curses = config.live or config.start_tab in {"live", "sense-radar", "capture"} or config.startup_intelligence or config.auto_baseline_policy != "off" or config.startup_suite_enabled
    if startup_visible_in_curses and backend != "textual":
        return _run_curses_tui(config)
    if backend != "curses" and _textual_available():
        return _run_textual_tui(config)
    return _run_curses_tui(config)


def _textual_available() -> bool:
    try:
        import rich  # noqa: F401
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def _run_curses_tui(config: CaptureConfig) -> list[dict[str, object]]:
    from .tui import run_curses_tui

    return run_curses_tui(config)


def _run_textual_tui(config: CaptureConfig) -> list[dict[str, object]]:
    from textual.app import App, ComposeResult
    from textual.widgets import DataTable, Footer, Header, Static, TabPane, TabbedContent

    from .tui import load_project_scenes

    class DSenseTextualApp(App):
        CSS = """
        Screen { background: $surface; }
        #summary { height: 6; border: solid $primary; padding: 1; }
        DataTable { height: 1fr; }
        Static.panel { border: solid $secondary; padding: 1; }
        """
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("r", "refresh", "Refresh"),
            ("u", "refresh", "Update"),
            ("c", "capture", "Capture in curses fallback"),
        ]

        def __init__(self, app_config: CaptureConfig):
            super().__init__()
            self.config = app_config
            training = train_models(app_config.project_name)
            self.state = AppState(
                config=app_config,
                channels=scan_channels(groups=app_config.channel_groups),
                scenes=load_project_scenes(app_config.project_name),
                baseline=training.baseline,
                classifier=training.classifier,
            )
            self.results: list[dict[str, object]] = []

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(self._summary_text(), id="summary")
            with TabbedContent():
                with TabPane("Record"):
                    yield Static(self._record_text(), classes="panel")
                with TabPane("Scenes"):
                    table = DataTable(id="scenes")
                    table.add_columns("Scene", "Label", "Status", "Confidence")
                    for scene in self.state.scenes:
                        quality = scene.get("quality", {})
                        quality = quality if isinstance(quality, dict) else {}
                        table.add_row(
                            str(scene.get("scene_id", "?")),
                            str(scene.get("label", "?")),
                            "ok" if scene.get("accepted", False) else "review",
                            str(quality.get("confidence", "?")),
                        )
                    yield table
                with TabPane("Channels"):
                    table = DataTable(id="channels")
                    table.add_columns("ID", "Group", "Bit", "Rate", "Status")
                    for channel in self.state.channels:
                        table.add_row(
                            str(channel.get("id", "")),
                            str(channel.get("group", "")),
                            str(channel.get("bit", "")),
                            str(channel.get("rate_hz", "")),
                            "online" if channel.get("available") else "offline",
                        )
                    yield table
                with TabPane("Learn"):
                    yield Static(self._learn_text(), classes="panel")
                with TabPane("Council"):
                    yield Static(self._council_text(), classes="panel")
                with TabPane("Classify"):
                    yield Static(self._classify_text(), classes="panel")
                with TabPane("Evaluation"):
                    yield Static(self._evaluation_text(), classes="panel")
                with TabPane("Watcher"):
                    yield Static("Press c to use the full recording workflow in the curses fallback. Press r to refresh project state.", classes="panel")
                with TabPane("Orbiters"):
                    yield Static("Orbiter summaries remain available from the curses fallback in this release.", classes="panel")
                with TabPane("Transfer"):
                    yield Static("Transfer export remains available from the curses fallback in this release.", classes="panel")
                with TabPane("Validate"):
                    yield Static("Validation remains available from the curses fallback in this release.", classes="panel")
                with TabPane("Help"):
                    yield Static("Optional Textual dashboard is active. Use c for the full capture workflow, u/r to refresh, q to quit.", classes="panel")
            yield Footer()

        def action_refresh(self) -> None:
            self.state.channels = scan_channels(groups=self.config.channel_groups)
            self.state.scenes = load_project_scenes(self.config.project_name)
            self.query_one("#summary", Static).update(self._summary_text())

        def action_capture(self) -> None:
            self.exit(_run_curses_tui(self.config))

        def _summary_text(self) -> str:
            counts = summarize_scene_counts(self.state.scenes)
            groups = ",".join(self.config.channel_groups)
            return (
                f"dSense Observatory | project {self.config.project_name} | channels {groups}\n"
                f"scenes {len(self.state.scenes)} | baseline {counts['baseline']} | user {counts['user']} | other {counts['other']}\n"
                f"label {self.config.label} | {self.config.duration:g}s at {self.config.tick_hz} Hz"
            )

        def _record_text(self) -> str:
            return (
                f"Mode: {self.config.mode}\n"
                f"Windows: pre {self.config.pre_roll:g}s | action {self.config.action:g}s | post {self.config.post_roll:g}s\n"
                "Press c to launch the full real-time curses recorder."
            )

        def _learn_text(self) -> str:
            if self.state.baseline is None:
                return "Baseline model not trained."
            return f"Baseline scenes: {self.state.baseline.scene_count}\nThreshold: {self.state.baseline.threshold:g}"

        def _council_text(self) -> str:
            prefix = "Startup intelligence disabled for this session.\n\n" if not self.config.startup_intelligence else ""
            return prefix + "\n".join(council_summary_lines(load_intelligence_state(self.config.project_name)))

        def _classify_text(self) -> str:
            if self.state.classifier is None:
                return "Classifier not trained."
            return f"Trained scenes: {self.state.classifier.scene_count}\nLabels: {len(self.state.classifier.label_counts)}"

        def _evaluation_text(self) -> str:
            try:
                report = evaluate_project_job(self.config.project_name)
            except (OSError, ValueError):
                return "Evaluation unavailable."
            lines = ["Repeatability"]
            lines.extend(evaluation_repeatability_lines(report))
            lines.append("")
            lines.append("Labels needing more takes")
            lines.extend(f"{label:<24} {reason}" for label, reason in labels_needing_more_takes(report, 6))
            lines.append("")
            lines.append("Useful channels")
            lines.extend(useful_channel_lines(report, 6))
            return "\n".join(lines)

    app = DSenseTextualApp(config)
    result = app.run()
    return result if isinstance(result, list) else []
