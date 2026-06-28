from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    label: str
    duration: float
    description: str
    notes: str
    pre_roll: float = 0.0
    action: float | None = None
    post_roll: float = 0.0
    mode: str = "user"
    manual: bool = True
    workload: str | None = None

    @property
    def action_seconds(self) -> float:
        if self.action is not None:
            return self.action
        return max(0.0, self.duration - self.pre_roll - self.post_roll)

    @property
    def automatable(self) -> bool:
        return not self.manual


USER_INTERACTION_SCENARIOS = [
    Scenario("user_approach_from_left", 10, "Person walks into sensing area from the left side of the computer.", "Start outside the sensing area on the left, enter during the action window.", 2, 5, 3),
    Scenario("user_approach_from_right", 10, "Person walks into sensing area from the right side of the computer.", "Start outside the sensing area on the right, enter during the action window.", 2, 5, 3),
    Scenario("user_walk_left_to_right", 10, "Person crosses in front of the computer left-to-right.", "Walk past the computer at a natural pace during the action window.", 2, 5, 3),
    Scenario("user_walk_right_to_left", 10, "Person crosses in front of the computer right-to-left.", "Walk past the computer at a natural pace during the action window.", 2, 5, 3),
    Scenario("user_stand_near_computer", 10, "Person stands still near the computer during action window.", "Stand near the computer without touching it.", 2, 5, 3),
    Scenario("user_sit_down_at_computer", 12, "Person approaches and sits down at the computer.", "Approach and sit during the action window.", 2, 7, 3),
    Scenario("user_leave_computer", 10, "Person gets up and walks away.", "Start near the computer, then leave during the action window.", 2, 5, 3),
    Scenario("user_typing_short_burst", 10, "User types steadily during action window.", "Type normally and steadily during the action window.", 2, 5, 3),
    Scenario("user_typing_hard_burst", 10, "User types harder/faster than normal during action window.", "Type with higher force or pace than normal.", 2, 5, 3),
    Scenario("user_mouse_activity", 10, "User moves/clicks mouse during action window.", "Move and click the mouse naturally during the action window.", 2, 5, 3),
    Scenario("user_phone_near_left_side", 10, "Phone is brought near the left side of the computer.", "Move a phone near the left side during action, then hold it still briefly.", 2, 5, 3),
    Scenario("user_phone_near_right_side", 10, "Phone is brought near the right side of the computer.", "Move a phone near the right side during action, then hold it still briefly.", 2, 5, 3),
    Scenario("user_door_open_close", 12, "Door or nearby object opens/closes during action window.", "Open or close the nearby object once during action.", 2, 7, 3),
    Scenario("user_table_tap_light", 8, "Light tap/vibration near the computer.", "Tap the table lightly once or twice during action.", 2, 3, 3),
    Scenario("user_table_tap_heavy", 8, "Stronger tap/vibration near the computer.", "Tap firmly enough to create a controlled vibration, without risking hardware.", 2, 3, 3),
]


BASELINE_SCENARIOS = [
    Scenario("baseline_idle_quiet", 12, "Quiet room, no intentional user action.", "Leave the machine untouched during the recording.", 2, 7, 3, mode="baseline", manual=False),
    Scenario("baseline_idle_screen_on", 12, "Computer awake, screen on, no user interaction.", "Keep the screen awake and avoid input.", 2, 7, 3, mode="baseline", manual=False),
    Scenario("baseline_idle_after_startup", 12, "Machine recently booted or apps recently opened, but no active user interaction.", "Prepare this state before recording; do not interact during capture.", 2, 7, 3, mode="baseline", manual=False),
    Scenario("baseline_idle_after_warmup", 12, "Machine has been running for several minutes, no active interaction.", "Use after the system has settled.", 2, 7, 3, mode="baseline", manual=False),
    Scenario("baseline_idle_charging", 12, "Laptop plugged into power, no intentional user action.", "Plug in power before capture if applicable.", 2, 7, 3, mode="baseline", manual=False),
    Scenario("baseline_idle_battery", 12, "Laptop on battery, no intentional user action.", "Unplug power before capture if applicable.", 2, 7, 3, mode="baseline", manual=False),
    Scenario("baseline_idle_network_on", 12, "Normal network connected, no intentional user action.", "Keep normal networking on; avoid deliberate downloads/uploads.", 2, 7, 3, mode="baseline", manual=False),
    Scenario("baseline_idle_low_light_activity", 12, "Normal background conditions, minor room noise accepted.", "Do not create an intentional event; ordinary background variation is acceptable.", 2, 7, 3, mode="baseline", manual=False),
]


ACTIVITY_SCENARIOS = [
    Scenario("activity_cpu_light", 10, "Light CPU activity during action window.", "Automatically runs a conservative CPU workload only during action.", 2, 5, 3, mode="activity", manual=False, workload="cpu_light"),
    Scenario("activity_cpu_heavy", 10, "Heavier CPU activity during action window.", "Automatically runs bounded CPU worker threads only during action.", 2, 5, 3, mode="activity", manual=False, workload="cpu_heavy"),
    Scenario("activity_disk_stat_burst", 10, "Repeated filesystem metadata/stat calls during action window.", "Automatically stats local temp files only during action.", 2, 5, 3, mode="activity", manual=False, workload="disk_stat_burst"),
    Scenario("activity_disk_write_tempfile", 10, "Temporary file writes during action window, cleaned up afterward.", "Automatically writes bounded temporary files and cleans them up.", 2, 5, 3, mode="activity", manual=False, workload="disk_write_tempfile"),
    Scenario("activity_memory_allocate_release", 10, "Allocate and release a modest memory block during action window.", "Automatically allocates and releases modest memory chunks.", 2, 5, 3, mode="activity", manual=False, workload="memory_allocate_release"),
    Scenario("activity_python_loop", 10, "Deterministic Python compute loop during action window.", "Automatically runs a deterministic Python loop only during action.", 2, 5, 3, mode="activity", manual=False, workload="python_loop"),
    Scenario("activity_mixed_cpu_disk", 10, "Combined CPU and disk activity.", "Automatically combines bounded CPU and temp-file activity.", 2, 5, 3, mode="activity", manual=False, workload="mixed_cpu_disk"),
    Scenario("activity_noop_control", 10, "Automatic recording with no artificial workload, used as an automation control.", "Records the same timed automation path without artificial workload.", 2, 5, 3, mode="activity", manual=False, workload="noop"),
]


SCENARIO_GROUPS = {
    "user": USER_INTERACTION_SCENARIOS,
    "baseline": BASELINE_SCENARIOS,
    "activity": ACTIVITY_SCENARIOS,
}


def all_scenarios() -> list[Scenario]:
    return [scenario for scenarios in SCENARIO_GROUPS.values() for scenario in scenarios]


def scenario_by_label(label: str) -> Scenario | None:
    for scenario in all_scenarios():
        if scenario.label == label:
            return scenario
    return None
