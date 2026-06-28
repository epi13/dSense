from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .autotest import validate_dataset
from .baseline import BaselineModel, train_and_save_project_baseline
from .classifier import SceneClassifierModel, train_and_save_project_classifier
from .council import run_intelligence_update
from .manifest import project_path
from .models.evaluation import evaluate_project_scenes
from .transfer import export_transfer_bundle
from .watcher import run_watcher_scan


JobFunction = Callable[[Callable[[str], None], threading.Event], str | None]


@dataclass(frozen=True)
class TrainingResult:
    baseline: BaselineModel | None
    classifier: SceneClassifierModel | None


@dataclass
class TuiJob:
    id: str
    name: str
    status: str = "idle"
    detail: str = ""
    started_at: float | None = None
    ended_at: float | None = None
    cancel_requested: bool = False
    error: str = ""

    @property
    def duration_s(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)


class TuiJobManager:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self._jobs: list[TuiJob] = []
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start(self, name: str, func: JobFunction) -> TuiJob:
        job = TuiJob(id=uuid.uuid4().hex[:10], name=name, status="run", started_at=time.monotonic())
        cancel_event = threading.Event()
        with self._lock:
            self._jobs.append(job)
            self._cancel[job.id] = cancel_event
        self._log(job, "started")

        def update(detail: str) -> None:
            with self._lock:
                job.detail = detail
            self._log(job, "progress")

        def run() -> None:
            try:
                detail = func(update, cancel_event)
                with self._lock:
                    job.ended_at = time.monotonic()
                    job.status = "cancelled" if cancel_event.is_set() else "done"
                    if detail:
                        job.detail = detail
            except Exception as exc:
                with self._lock:
                    job.ended_at = time.monotonic()
                    job.status = "error"
                    job.error = str(exc)
                    job.detail = str(exc)
            finally:
                self._log(job, job.status)

        thread = threading.Thread(target=run, name=f"dsense-{name.replace(' ', '-')}", daemon=True)
        thread.start()
        return job

    def snapshot(self) -> list[TuiJob]:
        with self._lock:
            return [TuiJob(**asdict(job)) for job in self._jobs]

    def add_completed(self, name: str, detail: str, status: str = "done", duration_s: float = 0.0) -> TuiJob:
        now = time.monotonic()
        job = TuiJob(
            id=uuid.uuid4().hex[:10],
            name=name,
            status=status,
            detail=detail,
            started_at=now - max(0.0, duration_s),
            ended_at=now,
        )
        with self._lock:
            self._jobs.append(job)
        self._log(job, status)
        return job

    def cancel_running(self) -> bool:
        with self._lock:
            running = [job for job in self._jobs if job.status == "run"]
            if not running:
                return False
            job = running[-1]
            job.cancel_requested = True
            job.detail = "cancel requested"
            event = self._cancel.get(job.id)
        if event is not None:
            event.set()
        self._log(job, "cancel_requested")
        return True

    def _log(self, job: TuiJob, event: str) -> None:
        path = project_path(self.project_name) / "jobs" / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(job)
        payload["event"] = event
        payload["duration_s"] = round(job.duration_s, 3)
        payload["logged_at"] = time.time()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def train_baseline(project_name: str) -> BaselineModel | None:
    try:
        model = train_and_save_project_baseline(project_name)
    except (OSError, ValueError):
        return None
    return model if model.scene_count else None


def train_classifier(project_name: str) -> SceneClassifierModel | None:
    try:
        model = train_and_save_project_classifier(project_name)
    except (OSError, ValueError):
        return None
    return model if model.scene_count else None


def train_models(project_name: str) -> TrainingResult:
    return TrainingResult(
        baseline=train_baseline(project_name),
        classifier=train_classifier(project_name),
    )


def validate_project(project_name: str) -> tuple[object, str]:
    result = validate_dataset(project_name)
    summary = f"{result.valid_scenes}/{result.total_scenes} valid, {result.error_count} errors, {result.warning_count} warnings"
    return result, summary


def run_watcher_job(project_name: str, channel_groups: list[str] | tuple[str, ...], duration: float = 3.0, tick_hz: int = 50) -> str:
    result = run_watcher_scan(project_name, duration=duration, tick_hz=tick_hz, channel_groups=channel_groups)
    detected = len(result.get("detected", []))
    scene = dict(result.get("scene", {}))
    return f"watcher saved {scene.get('scene_id', '?')} with {detected} auto events"


def export_transfer_job(project_name: str) -> str:
    bundle = export_transfer_bundle(project_name)
    return f"exported {bundle.get('total_scenes', 0)} scenes"


def evaluate_project_job(project_name: str) -> dict[str, object]:
    return evaluate_project_scenes(project_name)


def update_intelligence_job(
    project_name: str,
    *,
    run_watchers: bool = True,
    run_orbiters: bool = True,
    run_training: bool = True,
    run_transfer: bool = True,
) -> str:
    latest = ""

    def progress(update: dict[str, object]) -> None:
        nonlocal latest
        step = dict(update.get("step", {}))
        latest = f"{step.get('name', 'step')} {step.get('status', '')}".strip()

    state = run_intelligence_update(
        project_name,
        startup=False,
        run_watchers=run_watchers,
        run_orbiters=run_orbiters,
        run_training=run_training,
        run_transfer=run_transfer,
        progress_callback=progress,
    )
    council = dict(state.get("council", {}))
    return f"{state.get('status')} agreement={council.get('agreement')} confidence={council.get('overall_confidence')} {latest}"


def project_scene_root(project_name: str):
    return project_path(project_name) / "scenes"
