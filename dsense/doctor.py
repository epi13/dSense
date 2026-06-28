from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass

from .manifest import DEFAULT_PROJECT, project_path, scan_channels


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def run_doctor(project_name: str = DEFAULT_PROJECT) -> list[DoctorCheck]:
    checks = [
        _python_check(),
        _dataset_check(project_name),
        _terminal_check(),
        _permission_check(project_name),
    ]
    checks.extend(_channel_checks())
    return checks


def doctor_ok(checks: list[DoctorCheck]) -> bool:
    return all(check.status != "error" for check in checks)


def print_doctor_report(checks: list[DoctorCheck]) -> None:
    print("dSense doctor")
    for check in checks:
        print(f"{check.status.upper():<7} {check.name:<24} {check.detail}")


def _python_check() -> DoctorCheck:
    version = sys.version_info
    status = "ok" if version >= (3, 11) else "error"
    detail = f"Python {platform.python_version()} at {sys.executable}"
    if status == "error":
        detail += " (requires Python 3.11+)"
    return DoctorCheck("python", status, detail)


def _dataset_check(project_name: str) -> DoctorCheck:
    root = project_path(project_name)
    if not root.exists():
        return DoctorCheck("dataset", "warning", f"{root} does not exist; run 'dsense init {project_name}'")
    missing = [name for name in ("manifest.json", "scenes", "exports") if not (root / name).exists()]
    if missing:
        return DoctorCheck("dataset", "warning", f"{root} is missing: {', '.join(missing)}")
    return DoctorCheck("dataset", "ok", str(root))


def _terminal_check() -> DoctorCheck:
    term = os.environ.get("TERM", "")
    if not sys.stdout.isatty():
        return DoctorCheck("terminal", "warning", "stdout is not a TTY; TUI may not render here")
    if not term or term == "dumb":
        return DoctorCheck("terminal", "warning", f"TERM={term or '<unset>'}; TUI support may be limited")
    return DoctorCheck("terminal", "ok", f"TERM={term}")


def _permission_check(project_name: str) -> DoctorCheck:
    root = project_path(project_name)
    target = root if root.exists() else root.parent
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return DoctorCheck("permissions", "error", f"cannot create {target}: {exc}")
    if not os.access(target, os.R_OK | os.W_OK):
        return DoctorCheck("permissions", "error", f"{target} is not readable and writable")
    return DoctorCheck("permissions", "ok", f"read/write available at {target}")


def _channel_checks() -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    try:
        channels = scan_channels()
    except Exception as exc:
        return [DoctorCheck("channels", "error", f"channel scan failed: {exc}")]
    available = 0
    for channel in channels:
        status = "ok" if channel.get("available") else "warning"
        if status == "ok":
            available += 1
        detail = str(channel.get("reason", "ok"))
        checks.append(DoctorCheck(f"channel:{channel.get('id', '?')}", status, detail))
    if not available:
        checks.insert(0, DoctorCheck("channels", "error", "no channels are available"))
    else:
        checks.insert(0, DoctorCheck("channels", "ok", f"{available}/{len(channels)} available"))
    return checks
