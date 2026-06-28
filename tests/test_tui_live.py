from __future__ import annotations

import json

from dsense.tui_live import (
    LiveObservation,
    LiveSessionWriter,
    build_live_observation,
    classify_live_interval,
    direction_hypothesis_from_labels,
    live_session_path,
    score_unknown_anomaly,
    unknown_anomalies_from_score,
)


def test_live_observation_object_creation():
    observation = LiveObservation(
        created_utc="2026-06-28T00:00:00Z",
        project_name="base",
        tick=123,
        elapsed_s=1.23,
        channel_values={"dt_ns": 100.0},
        channel_status={"dt_ns": "ok"},
        baseline_score=0.12,
        classifier_label="baseline_idle",
        classifier_confidence=0.72,
        timeseries_label="baseline_idle",
        timeseries_confidence=0.69,
        watcher_score=0.31,
        council_confidence=0.7,
        council_agreement="high",
        proximity_hypothesis={"status": "normal", "direction": "unknown", "strength": 0.12, "label_hint": None, "confidence": 0.0},
    )

    assert observation.to_dict()["project_name"] == "base"
    assert observation.interval_classification == "normal"


def test_unknown_anomaly_scoring_weights_disagreement_and_low_confidence():
    low = score_unknown_anomaly(
        baseline_score=0.1,
        watcher_score=0.1,
        classifier_confidence=0.9,
        timeseries_confidence=0.9,
        channel_volatility=0.0,
        council_agreement="high",
    )
    high = score_unknown_anomaly(
        baseline_score=9.0,
        watcher_score=0.8,
        classifier_confidence=0.2,
        timeseries_confidence=0.2,
        channel_volatility=0.7,
        council_agreement="low",
    )

    assert low < 0.3
    assert high > 0.6


def test_known_vs_unknown_interval_classification():
    unknown = unknown_anomalies_from_score(0.75, classifier_confidence=0.2, timeseries_confidence=0.2, council_agreement="low")

    assert classify_live_interval([], unknown, 0.75, "low") == "needs_recording"
    assert classify_live_interval([{"name": "timing_spike"}], [], 0.2, "high") == "known_anomaly"
    assert classify_live_interval([], [], 0.1, "high") == "normal"


def test_direction_hypothesis_extracts_weak_direction_from_labels():
    east = direction_hypothesis_from_labels(["person_left_to_right"])
    unknown = direction_hypothesis_from_labels(["baseline_idle"])

    assert "east" in east["direction"] or "right" in east["direction"]
    assert east["status"].startswith("weak")
    assert unknown["direction"] == "unknown"
    assert "no trained directional scenes" in unknown["detail"]


def test_live_session_writer_throttles_and_writes_forced_events(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    observation = LiveObservation(
        created_utc="2026-06-28T00:00:00Z",
        project_name="base",
        tick=1,
        elapsed_s=1.0,
        channel_values={},
        channel_status={},
        baseline_score=0.0,
        classifier_label="unknown",
        classifier_confidence=0.0,
        timeseries_label="unknown",
        timeseries_confidence=0.0,
        watcher_score=0.0,
        council_confidence=0.0,
        council_agreement="unknown",
        proximity_hypothesis={},
    )
    writer = LiveSessionWriter("base", min_interval_s=60.0)

    assert writer.maybe_write(observation) is True
    assert writer.maybe_write(observation) is False
    assert writer.maybe_write(observation, force=True, event="user_mark_interval") is True

    rows = [json.loads(line) for line in live_session_path("base").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[-1]["event"] == "user_mark_interval"


def test_build_live_observation_surfaces_unknown_anomaly_without_claiming_presence():
    observation = build_live_observation(
        "base",
        tick=4,
        elapsed_s=0.4,
        channel_values={"dt_ns": 1_000_000.0, "sleep_drift_ns": 9000.0},
        channel_status={"dt_ns": "ok", "sleep_drift_ns": "ok"},
        recent_rows=[{"dt_ns": 1.0, "sleep_drift_ns": value} for value in [1.0, 2.0, 100.0, 2.0]],
        baseline=None,
        classifier=None,
        timeseries=None,
        council_state=None,
        watcher_events=[],
    )

    assert observation.council_agreement == "unknown"
    assert observation.proximity_hypothesis["direction"] == "unknown"
    assert "human" not in json.dumps(observation.to_dict()).lower()
