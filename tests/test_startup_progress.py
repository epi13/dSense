from __future__ import annotations

from dsense.council import run_intelligence_update
from dsense import cli
from dsense.cli import build_parser
from dsense.manifest import DEFAULT_PROJECT, init_project
from dsense.orbiters import orbiters_state_path, update_project_orbiters_incremental
from dsense.startup_progress import StartupStepProgress, make_progress
from dsense.tui import _startup_progress_line


def test_startup_progress_event_model():
    row = StartupStepProgress(name="orbiters", label="Orbiters", status="running", progress=None, message="working")

    assert row.to_dict()["name"] == "orbiters"
    assert make_progress("orbiters", "done", progress=1.0)["label"] == "Orbiters"


def test_progress_callback_receives_each_startup_step(sample_dataset):
    events = []

    state = run_intelligence_update(
        DEFAULT_PROJECT,
        startup=True,
        run_watchers=False,
        run_orbiters=False,
        run_transfer=False,
        progress_callback=events.append,
    )

    names = {event["name"] for event in events}
    assert {"init_project", "validate", "train_baseline", "train_classifier", "train_timeseries", "evaluate", "watcher", "orbiters", "transfer", "write_state"} <= names
    assert not any(step["status"] == "running" for step in state["steps"])


def test_startup_pipeline_continues_if_orbiters_fail(sample_dataset, monkeypatch):
    def fail_orbiters(*args, **kwargs):
        raise ValueError("orbiter stalled")

    monkeypatch.setattr("dsense.council.update_project_orbiters_incremental", fail_orbiters)

    state = run_intelligence_update(DEFAULT_PROJECT, startup=True, run_watchers=False, run_transfer=False)

    assert any(step["name"] == "orbiters" and step["status"] == "failed" for step in state["steps"])
    assert any(step["name"] == "write_state" and step["status"] == "ok" for step in state["steps"])


def test_startup_pipeline_can_skip_orbiters(sample_dataset):
    state = run_intelligence_update(DEFAULT_PROJECT, startup=True, run_watchers=False, run_transfer=False, skip_steps={"orbiters"})

    assert any(step["name"] == "orbiters" and step["status"] == "skipped" for step in state["steps"])


def test_orbiters_report_skipped_and_up_to_date(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_project(DEFAULT_PROJECT)

    skipped = update_project_orbiters_incremental(DEFAULT_PROJECT)
    again = update_project_orbiters_incremental(DEFAULT_PROJECT)

    assert skipped["status"] == "skipped"
    assert skipped["skipped_reason"] == "skipped: no watcher events"
    assert again["skipped_reason"] == "skipped: no watcher events"
    assert orbiters_state_path(DEFAULT_PROJECT).exists()


def test_startup_progress_rendering_known_and_unknown_progress():
    known = _startup_progress_line(make_progress("validate", "running", progress=0.5, elapsed_s=1.2, message="checking"), 0)
    unknown = _startup_progress_line(make_progress("orbiters", "running", progress=None, elapsed_s=5.7, message="scanning recent watcher events"), 1)

    assert "50%" in known
    assert "checking" in known
    assert "working" in unknown
    assert "scanning recent watcher events" in unknown


def test_update_intelligence_cli_prints_progress(sample_dataset, capsys):
    args = build_parser().parse_args(["update-intelligence", DEFAULT_PROJECT, "--no-watchers", "--no-orbiters", "--no-transfer"])

    cli.cmd_update_intelligence(args)

    out = capsys.readouterr().out
    assert "[1/" in out
    assert "Project Init" in out
    assert "Council" in out
