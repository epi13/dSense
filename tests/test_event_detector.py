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
