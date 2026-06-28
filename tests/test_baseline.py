import csv

from dsense.baseline import (
    default_auto_baseline_policy,
    ensure_startup_baseline,
    load_project_baseline,
    project_has_usable_baseline,
    score_against_baseline,
    train_and_save_project_baseline,
)
from dsense.cli import build_parser
from dsense.manifest import DEFAULT_PROJECT, init_project
from dsense.utils.files import read_json
from dsense.utils.files import write_json


def _write_baseline_scene(root, scene_id, values):
    scene_dir = root / "scenes" / scene_id
    scene_dir.mkdir()
    write_json(scene_dir / "scene.json", {"scene_id": scene_id, "label": "baseline_idle", "accepted": True})
    with (scene_dir / "preview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags"])
        writer.writeheader()
        for tick, drift in enumerate(values):
            writer.writerow({"tick": tick, "t_ns": tick, "dt_ns": 10_000_000, "sleep_drift_ns": drift, "process_ns_estimate": 8_000, "quality_flags": 0})


def test_train_and_score_baseline_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    _write_baseline_scene(root, "scene_000001", [100, 110, 90, 100, 105, 95])

    model = train_and_save_project_baseline(DEFAULT_PROJECT)
    loaded = load_project_baseline(DEFAULT_PROJECT)
    score = score_against_baseline({"sleep_drift_ns": 5000}, loaded)

    assert model.scene_count == 1
    assert loaded is not None
    assert "sleep_drift_ns" in loaded.channels
    assert "sleep_drift_ns_slope" in loaded.feature_manifest["features"]
    assert score["status"] == "anomaly"


def test_ensure_startup_baseline_records_new_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = ensure_startup_baseline(DEFAULT_PROJECT, duration=0.05, tick_hz=10, policy="missing-only")

    assert result["status"] == "recorded"
    assert project_has_usable_baseline(DEFAULT_PROJECT)
    scene = read_json(tmp_path / "datasets" / DEFAULT_PROJECT / "scenes" / result["scene_id"] / "scene.json")
    assert scene["label"] == "baseline_startup_auto"
    assert scene["mode"] == "baseline_auto"
    assert (tmp_path / "datasets" / DEFAULT_PROJECT / "exports" / "baseline_model.json").exists()


def test_default_auto_baseline_policy_uses_startup_on_linux(monkeypatch):
    import dsense.baseline

    monkeypatch.setattr(dsense.baseline.platform, "system", lambda: "Linux")

    assert default_auto_baseline_policy() == "startup"


def test_ensure_startup_baseline_off_policy_skips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = ensure_startup_baseline(DEFAULT_PROJECT, duration=0.05, tick_hz=10, policy="off")

    assert result["status"] == "skipped"
    assert not list((tmp_path / "datasets" / DEFAULT_PROJECT / "scenes").glob("scene_*"))


def test_ensure_startup_baseline_missing_only_reuses_existing_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = ensure_startup_baseline(DEFAULT_PROJECT, duration=0.05, tick_hz=10, policy="missing-only")

    second = ensure_startup_baseline(DEFAULT_PROJECT, duration=0.05, tick_hz=10, policy="missing-only")

    scenes = list((tmp_path / "datasets" / DEFAULT_PROJECT / "scenes").glob("scene_*"))
    assert first["status"] == "recorded"
    assert second["status"] == "reused"
    assert len(scenes) == 1


def test_ensure_startup_baseline_startup_policy_records_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_startup_baseline(DEFAULT_PROJECT, duration=0.05, tick_hz=10, policy="missing-only")

    second = ensure_startup_baseline(DEFAULT_PROJECT, duration=0.05, tick_hz=10, policy="startup")

    scenes = list((tmp_path / "datasets" / DEFAULT_PROJECT / "scenes").glob("scene_*"))
    assert second["status"] == "recorded"
    assert len(scenes) == 2


def test_ensure_startup_baseline_force_records_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_startup_baseline(DEFAULT_PROJECT, duration=0.05, tick_hz=10, policy="missing-only")

    forced = ensure_startup_baseline(DEFAULT_PROJECT, duration=0.05, tick_hz=10, policy="missing-only", force=True)

    scenes = list((tmp_path / "datasets" / DEFAULT_PROJECT / "scenes").glob("scene_*"))
    assert forced["status"] == "recorded"
    assert len(scenes) == 2


def test_ensure_startup_baseline_handles_unavailable_channels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class UnavailableChannel:
        id = "unavailable_test"
        name = "Unavailable Test"
        group = "linux"
        rate_hz = 1
        bit = 12

        def available(self):
            return False

        def start(self):
            return None

        def stop(self):
            return None

        def sample(self, tick, now_ns):
            raise AssertionError("unavailable channel should not be sampled")

    import dsense.recorder

    monkeypatch.setattr(dsense.recorder, "default_channels", lambda groups=None: [UnavailableChannel()])

    result = ensure_startup_baseline(DEFAULT_PROJECT, duration=0.05, tick_hz=10, policy="startup")

    assert result["status"] == "recorded"
    assert project_has_usable_baseline(DEFAULT_PROJECT)


def test_tui_parser_accepts_auto_baseline_flags():
    args = build_parser().parse_args([
        "tui",
        "base",
        "--no-auto-baseline",
        "--auto-baseline-policy",
        "startup",
        "--auto-baseline-duration",
        "10",
        "--force-auto-baseline",
    ])

    assert args.no_auto_baseline is True
    assert args.auto_baseline_policy == "startup"
    assert args.auto_baseline_duration == 10
    assert args.force_auto_baseline is True
