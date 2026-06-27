from dsense.manifest import DEFAULT_PROJECT, init_project, scan_channels, allocate_scene_id


def test_init_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project("demo")
    assert (root / "manifest.json").exists()
    assert (root / "channels.json").exists()
    assert (root / "scenes").is_dir()
    assert allocate_scene_id("demo") == "scene_000001"


def test_allocate_scene_id_skips_existing_scene_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = init_project(DEFAULT_PROJECT)
    (root / "scenes" / "scene_000001").mkdir()
    assert allocate_scene_id(DEFAULT_PROJECT) == "scene_000002"


def test_scan_channels():
    ids = {c["id"] for c in scan_channels()}
    assert {"clock_delta", "sleep_jitter", "process_probe"} <= ids
