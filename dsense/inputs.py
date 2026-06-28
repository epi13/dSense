from __future__ import annotations


def validate_capture_params(
    duration: float,
    tick_hz: int,
    pre_roll: float = 0.0,
    action: float | None = None,
    post_roll: float = 0.0,
    repeat: int = 1,
) -> None:
    if int(tick_hz) < 1:
        raise ValueError("tick_hz must be >= 1")
    if float(duration) <= 0:
        raise ValueError("duration must be positive")
    if int(repeat) < 1:
        raise ValueError("repeat must be >= 1")
    if float(pre_roll) < 0:
        raise ValueError("pre_roll must be >= 0")
    if action is not None and float(action) < 0:
        raise ValueError("action must be >= 0")
    if float(post_roll) < 0:
        raise ValueError("post_roll must be >= 0")
    action_seconds = float(action) if action is not None else max(0.0, float(duration) - float(pre_roll) - float(post_roll))
    window_total = float(pre_roll) + action_seconds + float(post_roll)
    if window_total - float(duration) > 1e-9:
        raise ValueError("pre_roll + action + post_roll must not exceed duration")


def validate_duration_windows(duration: float | None, pre_roll: float, action: float, post_roll: float) -> float:
    for name, value in {"pre_roll": pre_roll, "action": action, "post_roll": post_roll}.items():
        if float(value) < 0:
            raise ValueError(f"{name} must be >= 0")
    computed = float(pre_roll) + float(action) + float(post_roll)
    if duration is None:
        duration = computed
    if float(duration) <= 0:
        raise ValueError("duration must be positive")
    if abs(float(duration) - computed) > 1e-9:
        raise ValueError("duration must equal pre_roll + action + post_roll")
    return float(duration)
