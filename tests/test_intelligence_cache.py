from __future__ import annotations

import json

from dsense.council import run_intelligence_update
from dsense.manifest import DEFAULT_PROJECT
from dsense.models.scene_store import build_or_load_feature_store, dataset_fingerprint, feature_store_path
from dsense.perf import resolve_worker_count, startup_profile_path
from dsense.utils.files import read_json, write_json


def test_dataset_fingerprint_changes_for_label_accepted_and_preview(sample_dataset):
    original = dataset_fingerprint(DEFAULT_PROJECT)
    scene_path = sample_dataset / "scenes" / "scene_000001" / "scene.json"
    scene = read_json(scene_path)
    scene["label"] = "baseline_changed"
    write_json(scene_path, scene)
    changed_label = dataset_fingerprint(DEFAULT_PROJECT)
    assert changed_label["hash"] != original["hash"]

    scene["label"] = "baseline_idle"
    scene["accepted"] = False
    write_json(scene_path, scene)
    changed_accepted = dataset_fingerprint(DEFAULT_PROJECT)
    assert changed_accepted["hash"] != original["hash"]

    scene["accepted"] = True
    write_json(scene_path, scene)
    preview = sample_dataset / "scenes" / "scene_000001" / "preview.csv"
    preview.write_text(preview.read_text(encoding="utf-8") + "4,5,6,7,8,0,9\n", encoding="utf-8")
    changed_preview = dataset_fingerprint(DEFAULT_PROJECT)
    assert changed_preview["hash"] != original["hash"]


def test_dataset_fingerprint_ignores_unrelated_exports(sample_dataset):
    original = dataset_fingerprint(DEFAULT_PROJECT)
    (sample_dataset / "exports" / "scratch.json").write_text(json.dumps({"noise": True}), encoding="utf-8")
    assert dataset_fingerprint(DEFAULT_PROJECT)["hash"] == original["hash"]


def test_cached_artifacts_are_loaded_and_force_rebuilds(sample_dataset, monkeypatch):
    first = run_intelligence_update(DEFAULT_PROJECT, run_watchers=False, run_orbiters=False, run_transfer=False, workers=1)
    assert not any(dict(step).get("status") == "failed" for step in first["steps"])

    def fail_train(*args, **kwargs):
        raise AssertionError("cache should have been used")

    monkeypatch.setattr("dsense.council.train_project_baseline_from_store", fail_train)
    monkeypatch.setattr("dsense.council.train_project_classifier_from_store", fail_train)
    monkeypatch.setattr("dsense.council.train_project_timeseries_from_store", fail_train)
    monkeypatch.setattr("dsense.council.train_project_contrastive_from_store", fail_train)
    cached = run_intelligence_update(DEFAULT_PROJECT, run_watchers=False, run_orbiters=False, run_transfer=False, workers=1)
    summaries = {step["name"]: dict(step["summary"]) for step in cached["steps"]}
    assert summaries["train_baseline"]["cache_hit"] is True
    assert summaries["train_classifier"]["cache_hit"] is True
    assert summaries["train_timeseries"]["cache_hit"] is True
    assert summaries["train_contrastive"]["cache_hit"] is True
    assert summaries["evaluate"]["cache_hit"] is True

    forced = run_intelligence_update(DEFAULT_PROJECT, run_watchers=False, run_orbiters=False, run_transfer=False, force_update=True, workers=1)
    assert any(step["name"] == "train_baseline" and step["status"] == "failed" for step in forced["steps"])


def test_feature_store_reads_each_preview_once_and_reuses_cache(sample_dataset, monkeypatch):
    calls = []
    original = "dsense.models.scene_store.read_numeric_preview_rows"

    import dsense.models.scene_store as scene_store

    real_reader = scene_store.read_numeric_preview_rows

    def counting_reader(path):
        calls.append(path)
        return real_reader(path)

    monkeypatch.setattr(original, counting_reader)
    store = build_or_load_feature_store(DEFAULT_PROJECT, workers=1, force=True)
    assert len(calls) == store.scene_count
    assert feature_store_path(DEFAULT_PROJECT).exists()

    calls.clear()
    cached = build_or_load_feature_store(DEFAULT_PROJECT, workers=1)
    assert cached.cache_hit is True
    assert calls == []


def test_update_without_changes_skips_expensive_steps_and_writes_profile(sample_dataset):
    run_intelligence_update(DEFAULT_PROJECT, run_watchers=False, run_orbiters=False, run_transfer=False, workers=1)
    state = run_intelligence_update(DEFAULT_PROJECT, run_watchers=False, run_orbiters=False, run_transfer=False, workers=1)
    summaries = {step["name"]: dict(step["summary"]) for step in state["steps"]}

    assert summaries["feature_store"]["cache_hit"] is True
    assert summaries["train_baseline"]["cache_hit"] is True
    assert summaries["evaluate"]["cache_hit"] is True
    assert startup_profile_path(DEFAULT_PROJECT).exists()
    assert dict(state["startup_profile"]).get("slowest_step")


def test_worker_count_env_and_explicit(monkeypatch):
    monkeypatch.setenv("DSENSE_WORKERS", "3")
    assert resolve_worker_count() == 3
    assert resolve_worker_count(2) == 2
    monkeypatch.setenv("DSENSE_WORKERS", "bad")
    assert resolve_worker_count() >= 1
