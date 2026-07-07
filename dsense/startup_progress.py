from __future__ import annotations

from dataclasses import asdict, dataclass
from time import monotonic

from .utils.timebase import utc_now_iso


@dataclass
class StartupStepProgress:
    name: str
    label: str
    status: str = "pending"
    progress: float | None = 0.0
    current: int | None = None
    total: int | None = None
    message: str = ""
    started_utc: str | None = None
    finished_utc: str | None = None
    elapsed_s: float = 0.0
    warning: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


STEP_LABELS = {
    "init_project": "Project Init",
    "validate": "Validation",
    "train_baseline": "Baseline",
    "train_classifier": "Classifier",
    "train_timeseries": "Time-series",
    "train_contrastive": "Contrastive",
    "load_models": "Load Models",
    "evaluate": "Evaluation",
    "watcher": "Watchers",
    "orbiters": "Orbiters",
    "transfer": "Transfer",
    "write_state": "Council",
}

OPTIONAL_STARTUP_STEPS = {"watcher", "orbiters", "transfer"}


def make_progress(
    name: str,
    status: str,
    *,
    progress: float | None = None,
    current: int | None = None,
    total: int | None = None,
    message: str = "",
    started_utc: str | None = None,
    finished_utc: str | None = None,
    elapsed_s: float = 0.0,
    warning: str | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return StartupStepProgress(
        name=name,
        label=STEP_LABELS.get(name, name.replace("_", " ").title()),
        status=status,
        progress=progress,
        current=current,
        total=total,
        message=message,
        started_utc=started_utc,
        finished_utc=finished_utc,
        elapsed_s=round(elapsed_s, 3),
        warning=warning,
        error=error,
    ).to_dict()


def progress_warning(name: str, elapsed_s: float) -> str | None:
    if name == "orbiters" and elapsed_s >= 30.0:
        return "orbiter step is taking longer than expected"
    if elapsed_s >= 10.0:
        return "still working"
    return None


def initial_progress_rows() -> dict[str, dict[str, object]]:
    return {
        name: make_progress(name, "pending", progress=0.0)
        for name in STEP_LABELS
    }


def mark_done(row: dict[str, object], message: str = "") -> dict[str, object]:
    out = dict(row)
    out.update({"status": "done", "progress": 1.0, "message": message or str(row.get("message", "")), "finished_utc": utc_now_iso()})
    return out


def elapsed_from(started_monotonic: float | None) -> float:
    return max(0.0, monotonic() - started_monotonic) if started_monotonic is not None else 0.0
