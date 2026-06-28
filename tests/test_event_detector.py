from dsense.event_detector import HeuristicEventDetector


def test_heuristic_event_detector_emits_spike_after_warmup():
    detector = HeuristicEventDetector(tick_hz=20, threshold=5.0, cooldown_ms=100)
    events = []

    for tick in range(20):
        events.extend(detector.update({
            "tick": tick,
            "elapsed_ms": tick * 50,
            "dt_ns": 50_000_000,
            "sleep_drift_ns": 100_000,
            "process_ns_estimate": 8_000,
        }))

    events.extend(detector.update({
        "tick": 21,
        "elapsed_ms": 1050,
        "dt_ns": 50_000_000,
        "sleep_drift_ns": 8_000_000,
        "process_ns_estimate": 8_000,
    }))

    assert events
    assert events[-1]["event"] == "heuristic_signal_spike"
    assert events[-1]["source"] == "heuristic"
    assert events[-1]["channel"] == "sleep_drift_ns"
    assert detector.state.status == "event"


def test_heuristic_event_detector_uses_cooldown():
    detector = HeuristicEventDetector(tick_hz=20, threshold=5.0, cooldown_ms=1000)

    for tick in range(20):
        detector.update({
            "tick": tick,
            "elapsed_ms": tick * 50,
            "dt_ns": 50_000_000,
            "sleep_drift_ns": 100_000,
            "process_ns_estimate": 8_000,
        })

    first = detector.update({
        "tick": 21,
        "elapsed_ms": 1050,
        "dt_ns": 50_000_000,
        "sleep_drift_ns": 8_000_000,
        "process_ns_estimate": 8_000,
    })
    second = detector.update({
        "tick": 22,
        "elapsed_ms": 1100,
        "dt_ns": 50_000_000,
        "sleep_drift_ns": 9_000_000,
        "process_ns_estimate": 8_000,
    })

    assert len(first) == 1
    assert second == []


def test_detector_watches_all_learned_baseline_channels_by_default():
    detector = HeuristicEventDetector(
        tick_hz=20,
        learned_baseline={
            "dt_ns": {"center": 50_000_000.0, "mad": 1_000_000.0},
            "disk_stat_latency_ns": {"center": 100.0, "mad": 10.0},
            "battery_percent": {"center": 92.0, "mad": 1.0},
        },
        threshold=5.0,
        warmup_samples=1,
    )

    events = detector.update({
        "tick": 1,
        "elapsed_ms": 50,
        "values": {
            "dt_ns": 50_000_000,
            "disk_stat_latency_ns": 1_000,
            "battery_percent": 92,
        },
    })

    assert events
    assert events[0]["channel"] == "disk_stat_latency_ns"
    assert detector.state.channel == "disk_stat_latency_ns"


def test_detector_can_filter_watched_channels():
    detector = HeuristicEventDetector(
        tick_hz=20,
        learned_baseline={
            "disk_stat_latency_ns": {"center": 100.0, "mad": 10.0},
            "cpu_load_ppm": {"center": 10_000.0, "mad": 1_000.0},
        },
        watched_channels=["cpu_load_ppm"],
        threshold=5.0,
        warmup_samples=1,
    )

    events = detector.update({
        "tick": 1,
        "elapsed_ms": 50,
        "disk_stat_latency_ns": 1_000,
        "cpu_load_ppm": 30_000,
    })

    assert events
    assert events[0]["channel"] == "cpu_load_ppm"
