"""
Shared scenario definitions for baseline generation.
"""

from dataclasses import dataclass


@dataclass
class BaselineScenario:
    """Configuration for a baseline scenario to record."""
    label: str
    duration: float  # seconds
    description: str
    notes: str


# Idle baseline scenarios
BASELINE_SCENARIOS = [
    BaselineScenario(
        label="baseline_idle_early_morning",
        duration=60,
        description="System just booted or woken, very cold, minimal services",
        notes="Early morning idle, system at baseline temperature",
    ),
    BaselineScenario(
        label="baseline_idle_daytime",
        duration=60,
        description="Mid-day idle, system warmed up, normal services running",
        notes="Daytime idle after prolonged operation",
    ),
    BaselineScenario(
        label="baseline_idle_evening",
        duration=60,
        description="Evening idle, system stable and warm",
        notes="Evening idle, system in steady state",
    ),
    BaselineScenario(
        label="baseline_idle_with_fan",
        duration=45,
        description="Idle with fan running (thermal management active)",
        notes="Idle with active cooling. Note the fan noise in substrate.",
    ),
    BaselineScenario(
        label="baseline_idle_with_network",
        duration=45,
        description="Idle but with periodic network activity (ping, sync)",
        notes="Idle with background network operations",
    ),
    BaselineScenario(
        label="baseline_idle_with_disk_activity",
        duration=45,
        description="Idle with occasional disk operations (logging, cache)",
        notes="Idle with light I/O activity",
    ),
    BaselineScenario(
        label="baseline_low_activity_text_editor",
        duration=30,
        description="Single text editor open, no typing, no mouse movement",
        notes="Minimal GUI application running, no input",
    ),
    BaselineScenario(
        label="baseline_low_activity_browser_static",
        duration=30,
        description="Browser with static page loaded, no scrolling, no input",
        notes="Idle webpage, browser consuming minimal CPU",
    ),
    BaselineScenario(
        label="baseline_high_performance_mode",
        duration=45,
        description="High performance power profile (no power saving)",
        notes="Power policy set to performance, expect higher frequency variance",
    ),
    BaselineScenario(
        label="baseline_power_saver_mode",
        duration=45,
        description="Power saver profile (aggressive throttling)",
        notes="Power policy set to battery saver or eco mode",
    ),
    BaselineScenario(
        label="baseline_quiet_environment",
        duration=30,
        description="Silent room, no ambient noise, no vibration",
        notes="Quiet environment. No background sounds or vibration.",
    ),
    BaselineScenario(
        label="baseline_with_ambient_noise",
        duration=30,
        description="Room with background noise (AC, traffic, music low)",
        notes="Ambient background noise present (AC, traffic, etc)",
    ),
    BaselineScenario(
        label="baseline_immediately_after_boot",
        duration=30,
        description="Capture immediately after system boot",
        notes="System freshly booted. Expect initialization patterns.",
    ),
    BaselineScenario(
        label="baseline_after_long_idle",
        duration=30,
        description="After 30+ minutes with no activity",
        notes="System fully settled after extended idle period",
    ),
    BaselineScenario(
        label="baseline_battery_low",
        duration=30,
        description="Laptop on battery with low charge (<20%)",
        notes="Low battery condition. Power limits may be active.",
    ),
    BaselineScenario(
        label="baseline_plugged_in",
        duration=30,
        description="Laptop plugged in, charging/charged",
        notes="Plugged to AC power, no battery constraint",
    ),
    BaselineScenario(
        label="baseline_external_display_connected",
        duration=30,
        description="External monitor/display connected and active",
        notes="External GPU/display active, driver in use",
    ),
]


# Activity scenarios
ACTIVITY_SCENARIOS = [
    BaselineScenario(
        label="baseline_cpu_light",
        duration=30,
        description="Light CPU activity (single core ~20%)",
        notes="Single core light computation running",
    ),
    BaselineScenario(
        label="baseline_cpu_moderate",
        duration=30,
        description="Moderate CPU activity (multi-core ~50%)",
        notes="Multi-threaded CPU load at ~50%",
    ),
    BaselineScenario(
        label="baseline_cpu_heavy",
        duration=30,
        description="Heavy CPU load (all cores ~90%+)",
        notes="Sustained high CPU utilization",
    ),
    BaselineScenario(
        label="baseline_memory_pressure",
        duration=30,
        description="High memory pressure (swap activity)",
        notes="Memory allocation causing swap activity",
    ),
    BaselineScenario(
        label="baseline_io_reads",
        duration=30,
        description="Sustained disk read activity",
        notes="Sequential disk reads from large file",
    ),
    BaselineScenario(
        label="baseline_io_writes",
        duration=30,
        description="Sustained disk write activity",
        notes="Sequential disk writes to file",
    ),
    BaselineScenario(
        label="baseline_io_random",
        duration=30,
        description="Random disk I/O pattern",
        notes="Random access I/O operations",
    ),
    BaselineScenario(
        label="baseline_network_download",
        duration=30,
        description="Network download activity",
        notes="Downloading file from local/remote source",
    ),
    BaselineScenario(
        label="baseline_network_upload",
        duration=30,
        description="Network upload activity",
        notes="Uploading file to local/remote destination",
    ),
    BaselineScenario(
        label="baseline_process_spawn",
        duration=30,
        description="Frequent process creation",
        notes="Spawning/terminating processes repeatedly",
    ),
    BaselineScenario(
        label="baseline_context_switch_heavy",
        duration=30,
        description="High context switch rate",
        notes="Many threads contending for CPU",
    ),
    BaselineScenario(
        label="baseline_interrupt_driven",
        duration=30,
        description="High interrupt/signal rate",
        notes="Frequent interrupts from devices",
    ),
]
