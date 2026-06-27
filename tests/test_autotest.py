"""Tests for the autotest validation module."""

import tempfile
import json
from pathlib import Path
from dsense.autotest import validate_scene, validate_dataset, ValidationError
from dsense.frame import build_frame, FRAME_SIZE
from dsense.utils.files import write_json


def create_test_scene(tmpdir: Path, scene_id: str, label: str, duration_ms: int = 10000, 
                      tick_hz: int = 100, include_error: str | None = None) -> Path:
    """Helper to create a minimal valid test scene."""
    scene_dir = tmpdir / "scenes" / scene_id
    scene_dir.mkdir(parents=True)
    
    expected_frames = max(1, int(round(duration_ms / 1000 * tick_hz)))
    
    # Create valid scene.json
    scene_data = {
        "scene_id": scene_id,
        "label": label,
        "duration_ms": duration_ms,
        "tick_hz": tick_hz,
        "frame_size_bytes": FRAME_SIZE,
        "quality": {
            "expected_frames": expected_frames,
            "actual_frames": expected_frames,
            "confidence": 0.99,
            "checksum_ok": include_error != "bad_checksum",
            "frame_size_valid": include_error != "bad_frame_size",
            "jitter_ns": 95000.0,
            "dropped_or_late_estimate": 0,
            "channel_availability_mask": 7,
        },
        "accepted": True,
    }
    write_json(scene_dir / "scene.json", scene_data)
    
    # Create valid frames.ds64
    frame_data = b""
    for i in range(expected_frames):
        frame = build_frame(i, 1000000000 + i * 10000000, 0b111, 0b010, 10000000, 0, 1000)
        frame_data += frame
    
    if include_error == "truncated_frames":
        # Truncate frame file
        frame_data = frame_data[:len(frame_data) // 2]
    
    (scene_dir / "frames.ds64").write_bytes(frame_data)
    
    # Create events.jsonl
    events = [
        {"t_ms": 0, "event": "scene_start"},
        {"t_ms": 2000, "event": "action_start"},
        {"t_ms": 7000, "event": "action_end"},
        {"t_ms": duration_ms, "event": "scene_end"},
    ]
    (scene_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    
    # Create preview.csv
    preview_header = "tick,t_ns,dt_ns,sleep_drift_ns,process_ns_estimate,quality_flags\n"
    preview_rows = ""
    for i in range(expected_frames):
        preview_rows += f"{i},1000000000,10000000,0,1000,2\n"
    (scene_dir / "preview.csv").write_text(preview_header + preview_rows)
    
    # Create checksum.txt
    import hashlib
    sha = hashlib.sha256(frame_data).hexdigest()
    if include_error == "bad_checksum":
        sha = "0" * 64  # Wrong checksum
    (scene_dir / "checksum.txt").write_text(f"sha256  frames.ds64  {sha}\n")
    
    # Create notes.txt
    (scene_dir / "notes.txt").write_text("")
    
    return scene_dir


def test_validate_scene_valid():
    """Test validation of a completely valid scene."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_scene(tmpdir_path, "scene_000001", "baseline_idle")
        
        result = validate_scene(tmpdir_path / "scenes" / "scene_000001")
        
        assert result.valid
        assert len([e for e in result.errors if e.severity == "error"]) == 0
        assert result.stats["label"] == "baseline_idle"
        assert result.stats["confidence"] == 0.99
        assert result.stats["checksum_verified"]


def test_validate_scene_missing_file():
    """Test validation catches missing required files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        scene_dir = create_test_scene(tmpdir_path, "scene_000001", "test")
        
        # Remove a required file
        (scene_dir / "checksum.txt").unlink()
        
        result = validate_scene(scene_dir)
        
        assert not result.valid
        errors = [e for e in result.errors if e.severity == "error"]
        assert len(errors) > 0
        assert any("checksum.txt" in e.message for e in errors)


def test_validate_scene_frame_truncation():
    """Test validation catches truncated frame file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_scene(tmpdir_path, "scene_000001", "test", include_error="truncated_frames")
        
        result = validate_scene(tmpdir_path / "scenes" / "scene_000001")
        
        assert not result.valid
        errors = [e for e in result.errors if e.severity == "error"]
        assert len(errors) > 0


def test_validate_scene_bad_checksum():
    """Test validation catches mismatched checksum."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_scene(tmpdir_path, "scene_000001", "test", include_error="bad_checksum")
        
        result = validate_scene(tmpdir_path / "scenes" / "scene_000001")
        
        assert not result.valid
        errors = [e for e in result.errors if e.severity == "error"]
        assert any("checksum" in e.check for e in errors)


def test_validate_dataset():
    """Test validation of complete dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / "demo_test").mkdir()
        
        # Create manifest
        manifest_data = {
            "project_name": "demo_test",
            "created_utc": "2026-06-27T00:00:00Z",
            "format": "dsense-scene-v0",
            "next_scene": 3,
        }
        write_json(tmpdir_path / "demo_test" / "manifest.json", manifest_data)
        
        # Create test scenes
        create_test_scene(tmpdir_path / "demo_test", "scene_000001", "baseline_idle", 30000)
        create_test_scene(tmpdir_path / "demo_test", "scene_000002", "walkby", 10000)
        
        # Mock the project_path to point to our test directory
        import dsense.manifest
        old_datasets = dsense.manifest.DATASETS
        dsense.manifest.DATASETS = tmpdir_path
        dsense.autotest.project_path = lambda name: tmpdir_path / name
        
        try:
            result = validate_dataset("demo_test")
            
            assert result.total_scenes == 2
            assert result.valid_scenes == 2
            assert result.error_count == 0
            assert result.comparison["unique_labels"] == 2
            assert result.comparison["total_frames"] == 4000  # 3000 + 1000
        finally:
            dsense.manifest.DATASETS = old_datasets


def test_validate_dataset_mixed_quality():
    """Test validation detects label inconsistencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / "demo_test").mkdir()
        
        # Create manifest
        write_json(tmpdir_path / "demo_test" / "manifest.json", {
            "project_name": "demo_test",
            "created_utc": "2026-06-27T00:00:00Z",
            "format": "dsense-scene-v0",
            "next_scene": 4,
        })
        
        # Create multiple scenes with same label
        create_test_scene(tmpdir_path / "demo_test", "scene_000001", "walkby")
        create_test_scene(tmpdir_path / "demo_test", "scene_000002", "walkby")
        create_test_scene(tmpdir_path / "demo_test", "scene_000003", "walkby")
        
        import dsense.manifest
        old_datasets = dsense.manifest.DATASETS
        dsense.manifest.DATASETS = tmpdir_path
        dsense.autotest.project_path = lambda name: tmpdir_path / name
        
        try:
            result = validate_dataset("demo_test")
            
            assert result.total_scenes == 3
            assert result.comparison["labels"]["walkby"] == 3
            assert result.comparison["unique_labels"] == 1
        finally:
            dsense.manifest.DATASETS = old_datasets
