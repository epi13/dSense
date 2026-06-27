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

    @property
    def action_seconds(self) -> float:
        if self.action is not None:
            return self.action
        return max(0.0, self.duration - self.pre_roll - self.post_roll)


BASELINE_SCENARIOS = [
    Scenario("baseline_idle_early_morning", 60, "System just booted or woken, cold, minimal services", "Early morning idle, system at baseline temperature"),
    Scenario("baseline_idle_daytime", 60, "Mid-day idle, warmed up, normal services running", "Daytime idle after prolonged operation"),
    Scenario("baseline_idle_evening", 60, "Evening idle, system stable and warm", "Evening idle, system in steady state"),
    Scenario("baseline_idle_with_fan", 45, "Idle with fan running", "Idle with active cooling. Note fan state."),
    Scenario("baseline_idle_with_network", 45, "Idle with periodic network activity", "Idle with background network operations"),
    Scenario("baseline_idle_with_disk_activity", 45, "Idle with occasional disk operations", "Idle with light I/O activity"),
    Scenario("baseline_low_activity_text_editor", 30, "Text editor open, no typing", "Minimal GUI app running, no input"),
    Scenario("baseline_low_activity_browser_static", 30, "Browser on static page, no input", "Idle webpage, browser consuming minimal CPU"),
    Scenario("baseline_high_performance_mode", 45, "High performance power profile", "Power policy set to performance"),
    Scenario("baseline_power_saver_mode", 45, "Power saver profile", "Power policy set to battery saver or eco mode"),
    Scenario("baseline_quiet_environment", 30, "Silent room, no ambient vibration", "Quiet environment. No background sounds or vibration."),
    Scenario("baseline_with_ambient_noise", 30, "Room with background noise", "Ambient background noise present"),
    Scenario("baseline_immediately_after_boot", 30, "Immediately after system boot", "System freshly booted. Expect initialization patterns."),
    Scenario("baseline_after_long_idle", 30, "After 30+ minutes idle", "System fully settled after extended idle period"),
    Scenario("baseline_battery_low", 30, "Laptop on low battery", "Low battery condition. Power limits may be active."),
    Scenario("baseline_plugged_in", 30, "Laptop plugged in", "Plugged to AC power, no battery constraint"),
    Scenario("baseline_external_display_connected", 30, "External monitor active", "External GPU/display active, driver in use"),
]


ACTIVITY_SCENARIOS = [
    Scenario("baseline_cpu_light", 30, "Light CPU activity", "Single core light computation running"),
    Scenario("baseline_cpu_moderate", 30, "Moderate CPU activity", "Multi-threaded CPU load at about 50%"),
    Scenario("baseline_cpu_heavy", 30, "Heavy CPU load", "Sustained high CPU utilization"),
    Scenario("baseline_memory_pressure", 30, "High memory pressure", "Memory allocation causing swap activity"),
    Scenario("baseline_io_reads", 30, "Sustained disk reads", "Sequential disk reads from a large file"),
    Scenario("baseline_io_writes", 30, "Sustained disk writes", "Sequential disk writes to a file"),
    Scenario("baseline_io_random", 30, "Random disk I/O", "Random access I/O operations"),
    Scenario("baseline_network_download", 30, "Network download activity", "Downloading file from local or remote source"),
    Scenario("baseline_network_upload", 30, "Network upload activity", "Uploading file to local or remote destination"),
    Scenario("baseline_process_spawn", 30, "Frequent process creation", "Spawning and terminating processes repeatedly"),
    Scenario("baseline_context_switch_heavy", 30, "High context switch rate", "Many threads contending for CPU"),
    Scenario("baseline_interrupt_driven", 30, "High interrupt/signal rate", "Frequent interrupts from devices"),
]


USER_INTERACTION_SCENARIOS = [
    Scenario("user_interaction_approach", 10, "User approaches the machine", "User starts away from the machine, approaches during action window", 2, 5, 3),
    Scenario("user_interaction_leave", 10, "User leaves the machine", "User starts near the machine, leaves during action window", 2, 5, 3),
    Scenario("person_walks_front_left_to_right", 10, "Walk-by left to right", "Person walks in front of the machine from left to right", 2, 5, 3),
    Scenario("person_walks_front_right_to_left", 10, "Walk-by right to left", "Person walks in front of the machine from right to left", 2, 5, 3),
    Scenario("user_sits_down_near_machine", 12, "User sits near machine", "User enters and sits down near the machine", 3, 6, 3),
    Scenario("phone_near_computer", 10, "Phone moved near computer", "Move phone near the machine during action window", 2, 5, 3),
    Scenario("door_open_close", 12, "Door opens or closes", "Door movement during action window", 3, 6, 3),
    Scenario("typing_burst", 10, "Typing burst", "User types during action window", 2, 5, 3),
    Scenario("mouse_activity", 10, "Mouse activity", "User moves/clicks mouse during action window", 2, 5, 3),
]


SCENARIO_GROUPS = {
    "user": USER_INTERACTION_SCENARIOS,
    "baseline": BASELINE_SCENARIOS,
    "activity": ACTIVITY_SCENARIOS,
}
