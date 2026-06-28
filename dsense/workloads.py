from __future__ import annotations

import math
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path


WorkloadFunc = Callable[[threading.Event], None]
ProgressCallback = Callable[[dict[str, object]], list[dict[str, object]] | None]


def cpu_light(stop_event: threading.Event) -> None:
    value = 0.0
    while not stop_event.is_set():
        for n in range(2_000):
            value += math.sin(n) * math.cos(n)
        stop_event.wait(0.01)
    _ = value


def cpu_heavy(stop_event: threading.Event) -> None:
    workers = []
    worker_count = max(1, min(2, (os.cpu_count() or 1)))
    for _ in range(worker_count):
        worker = threading.Thread(target=python_loop, args=(stop_event,), daemon=True)
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join(timeout=0.5)


def disk_stat_burst(stop_event: threading.Event, path: Path | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="dsense-stat-") as tmp:
        root = Path(path) if path is not None else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        files = []
        for index in range(8):
            item = root / f"probe-{index}.tmp"
            item.write_text("dsense\n", encoding="utf-8")
            files.append(item)
        while not stop_event.is_set():
            for item in files:
                try:
                    item.stat()
                except OSError:
                    pass
            stop_event.wait(0.005)


def disk_write_tempfile(stop_event: threading.Event) -> None:
    block = b"dsense-workload\n" * 128
    with tempfile.TemporaryDirectory(prefix="dsense-write-") as tmp:
        root = Path(tmp)
        index = 0
        while not stop_event.is_set():
            path = root / f"write-{index % 8}.tmp"
            with path.open("wb") as handle:
                for _ in range(16):
                    handle.write(block)
                    if stop_event.is_set():
                        break
            index += 1
            stop_event.wait(0.01)


def disk_read_tempfile(stop_event: threading.Event) -> None:
    block = b"dsense-read-workload\n" * 128
    with tempfile.TemporaryDirectory(prefix="dsense-read-") as tmp:
        path = Path(tmp) / "read.tmp"
        path.write_bytes(block * 16)
        while not stop_event.is_set():
            try:
                path.read_bytes()
            except OSError:
                pass
            stop_event.wait(0.01)


def memory_allocate_release(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        chunks = [bytearray(256 * 1024) for _ in range(8)]
        for index, chunk in enumerate(chunks):
            chunk[0] = index
        del chunks
        stop_event.wait(0.05)


def python_loop(stop_event: threading.Event) -> None:
    total = 0
    while not stop_event.is_set():
        for n in range(20_000):
            total = (total + (n * n + 17)) % 1_000_003
    _ = total


def mixed_cpu_disk(stop_event: threading.Event) -> None:
    disk_stop = threading.Event()
    disk_thread = threading.Thread(target=disk_write_tempfile, args=(disk_stop,), daemon=True)
    disk_thread.start()
    try:
        cpu_light(stop_event)
    finally:
        disk_stop.set()
        disk_thread.join(timeout=1.0)


def proc_read(stop_event: threading.Event) -> None:
    paths = [Path("/proc/stat"), Path("/proc/self/status"), Path("/proc/meminfo")]
    readable = [path for path in paths if path.exists()]
    while not stop_event.is_set():
        for path in readable:
            try:
                path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        stop_event.wait(0.02)


def sysfs_read(stop_event: threading.Event) -> None:
    roots = [Path("/sys/class/thermal"), Path("/sys/class/power_supply")]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.glob("*/*") if path.is_file())
    paths = paths[:16]
    while not stop_event.is_set():
        for path in paths:
            try:
                path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        stop_event.wait(0.05)


def memory_cpu(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        chunks = [bytearray(128 * 1024) for _ in range(4)]
        total = 0
        for n in range(5_000):
            total = (total + n * n) % 1_000_003
        del chunks
        _ = total
        stop_event.wait(0.02)


def noop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        stop_event.wait(0.05)


WORKLOADS: dict[str, WorkloadFunc] = {
    "cpu_light": cpu_light,
    "cpu_heavy": cpu_heavy,
    "disk_stat_burst": disk_stat_burst,
    "disk_write_tempfile": disk_write_tempfile,
    "disk_read_tempfile": disk_read_tempfile,
    "memory_allocate_release": memory_allocate_release,
    "python_loop": python_loop,
    "mixed_cpu_disk": mixed_cpu_disk,
    "proc_read": proc_read,
    "sysfs_read": sysfs_read,
    "memory_cpu": memory_cpu,
    "noop": noop,
}


def valid_workload_ids() -> set[str]:
    return set(WORKLOADS)


class TimedWorkload:
    def __init__(self, workload_id: str | None, action_start_ms: int, action_end_ms: int):
        self.workload_id = workload_id
        self.action_start_ms = max(0, action_start_ms)
        self.action_end_ms = max(self.action_start_ms, action_end_ms)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started = False
        self.stopped = False

    def update(self, elapsed_ms: int) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        if self.workload_id is None:
            return events
        if not self.started and elapsed_ms >= self.action_start_ms:
            self.started = True
            self._start()
            events.append({"event": "workload_start", "source": "workload", "detail": self.workload_id, "t_ms": elapsed_ms})
        if self.started and not self.stopped and elapsed_ms >= self.action_end_ms:
            self.stopped = True
            self._stop()
            events.append({"event": "workload_end", "source": "workload", "detail": self.workload_id, "t_ms": elapsed_ms})
        return events

    def finish(self) -> list[dict[str, object]]:
        if self.started and not self.stopped:
            self.stopped = True
            self._stop()
            return [{"event": "workload_end", "source": "workload", "detail": self.workload_id}]
        return []

    def _start(self) -> None:
        func = WORKLOADS.get(str(self.workload_id))
        if func is None:
            raise ValueError(f"Unknown workload id: {self.workload_id}")
        self.thread = threading.Thread(target=func, args=(self.stop_event,), daemon=True)
        self.thread.start()

    def _stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


def workload_progress_callback(
    workload_id: str | None,
    pre_roll: float,
    action: float,
    inner: ProgressCallback | None = None,
) -> ProgressCallback | None:
    if workload_id is None and inner is None:
        return None
    timed = TimedWorkload(
        workload_id,
        action_start_ms=int(pre_roll * 1000),
        action_end_ms=int((pre_roll + action) * 1000),
    )

    def progress(update: dict[str, object]) -> list[dict[str, object]]:
        elapsed_ms = int(update.get("elapsed_ms", 0))
        events = timed.update(elapsed_ms)
        if inner is not None:
            events.extend(inner(update) or [])
        expected = int(update.get("expected", 1))
        tick = int(update.get("tick", 0))
        if tick + 1 >= expected:
            end_events = timed.finish()
            duration_ms = int(update.get("duration_ms", elapsed_ms))
            for event in end_events:
                event.setdefault("t_ms", duration_ms)
            events.extend(end_events)
        return events

    return progress
