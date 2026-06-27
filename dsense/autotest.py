"""
Comprehensive autotest and validation suite for dSense datasets.

Validates scene integrity, frame data, checksums, metadata consistency,
and provides detailed quality reports and cross-scene comparisons.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frame import parse_frame, FRAME_SIZE, verify_frame
from .manifest import project_path
from .utils.files import read_json


@dataclass
class ValidationError:
    """Represents a validation error or warning."""
    scene_id: str
    severity: str  # 'error', 'warning', 'info'
    check: str
    message: str


@dataclass
class SceneValidationResult:
    """Results for a single scene validation."""
    scene_id: str
    label: str
    valid: bool
    errors: list[ValidationError]
    stats: dict[str, Any]


@dataclass
class DatasetValidationResult:
    """Results for entire dataset validation."""
    project_name: str
    total_scenes: int
    valid_scenes: int
    error_count: int
    warning_count: int
    scenes: list[SceneValidationResult]
    comparison: dict[str, Any]


def validate_scene(scene_path: Path) -> SceneValidationResult:
    """
    Validate a single scene for data integrity and consistency.
    
    Checks:
    - All required files exist
    - JSON schema validity
    - Frame file size and format
    - Checksum verification
    - Preview CSV alignment with frames
    - Event timing consistency
    - Metadata consistency
    """
    scene_id = scene_path.name
    errors: list[ValidationError] = []
    stats: dict[str, Any] = {}
    
    # Check required files
    required_files = ["scene.json", "frames.ds64", "events.jsonl", "preview.csv", "checksum.txt", "notes.txt"]
    for fname in required_files:
        fpath = scene_path / fname
        if not fpath.exists():
            errors.append(ValidationError(scene_id, "error", "file_exists", f"Missing required file: {fname}"))
    
    # Try to load scene.json
    scene_json_path = scene_path / "scene.json"
    if scene_json_path.exists():
        try:
            scene_data = read_json(scene_json_path)
            stats["label"] = scene_data.get("label", "unknown")
            stats["duration_ms"] = scene_data.get("duration_ms", 0)
            stats["tick_hz"] = scene_data.get("tick_hz", 100)
            stats["confidence"] = scene_data.get("quality", {}).get("confidence", 0)
            stats["expected_frames"] = scene_data.get("quality", {}).get("expected_frames", 0)
            stats["actual_frames"] = scene_data.get("quality", {}).get("actual_frames", 0)
            stats["checksum_ok"] = scene_data.get("quality", {}).get("checksum_ok", False)
            stats["frame_size_valid"] = scene_data.get("quality", {}).get("frame_size_valid", False)
            stats["jitter_ns"] = scene_data.get("quality", {}).get("jitter_ns", 0)
            stats["accepted"] = scene_data.get("accepted", False)
            
            # Validate metadata consistency
            if scene_data.get("scene_id") != scene_id:
                errors.append(ValidationError(scene_id, "error", "metadata", 
                    f"scene.json id mismatch: {scene_data.get('scene_id')} != {scene_id}"))
            
            if not scene_data.get("label"):
                errors.append(ValidationError(scene_id, "warning", "metadata", "Missing label"))
                
            if scene_data.get("frame_size_bytes") != FRAME_SIZE:
                errors.append(ValidationError(scene_id, "error", "metadata",
                    f"Frame size mismatch: {scene_data.get('frame_size_bytes')} != {FRAME_SIZE}"))
        except Exception as e:
            errors.append(ValidationError(scene_id, "error", "json_parse", f"Failed to parse scene.json: {e}"))
    
    # Validate frames.ds64
    frames_path = scene_path / "frames.ds64"
    if frames_path.exists():
        try:
            frame_data = frames_path.read_bytes()
            expected_size = stats.get("expected_frames", 0) * FRAME_SIZE
            stats["file_size_bytes"] = len(frame_data)
            
            # Check file size
            if len(frame_data) % FRAME_SIZE != 0:
                errors.append(ValidationError(scene_id, "error", "frame_size",
                    f"Frame file size {len(frame_data)} not divisible by frame size {FRAME_SIZE}"))
            else:
                frame_count = len(frame_data) // FRAME_SIZE
                stats["file_frame_count"] = frame_count
                
                if expected_size > 0 and len(frame_data) != expected_size:
                    errors.append(ValidationError(scene_id, "error", "frame_count",
                        f"Frame count mismatch: {frame_count} != {stats.get('expected_frames', 0)}"))
                
                # Sample verify frames (every 100th frame to avoid slowdown)
                bad_frames = 0
                sample_size = max(1, frame_count // 100) or 1
                for i in range(0, frame_count, sample_size):
                    frame_offset = i * FRAME_SIZE
                    frame = frame_data[frame_offset:frame_offset + FRAME_SIZE]
                    if not verify_frame(frame):
                        bad_frames += 1
                
                if bad_frames > 0:
                    errors.append(ValidationError(scene_id, "error", "frame_verify",
                        f"Frame checksum failures detected in {bad_frames} samples"))
                else:
                    stats["frames_verified"] = True
        except Exception as e:
            errors.append(ValidationError(scene_id, "error", "frame_read", f"Failed to read frames.ds64: {e}"))
    
    # Validate events.jsonl
    events_path = scene_path / "events.jsonl"
    if events_path.exists():
        try:
            events = []
            for line in events_path.read_text().splitlines():
                if line.strip():
                    events.append(json.loads(line))
            
            stats["event_count"] = len(events)
            expected_events = {"scene_start", "action_start", "action_end", "scene_end"}
            event_types = {e.get("event") for e in events}
            
            if not expected_events <= event_types:
                missing = expected_events - event_types
                if missing:
                    errors.append(ValidationError(scene_id, "error", "events",
                        f"Missing event types: {missing}"))
            
            # Check timing consistency
            if events:
                times = [e.get("t_ms", 0) for e in events]
                if times != sorted(times):
                    errors.append(ValidationError(scene_id, "error", "events",
                        "Event times not in order"))
                
                expected_duration = stats.get("duration_ms", 0)
                if events[-1].get("t_ms", 0) != expected_duration:
                    errors.append(ValidationError(scene_id, "warning", "events",
                        f"Last event time {events[-1].get('t_ms')} != duration {expected_duration}"))
        except Exception as e:
            errors.append(ValidationError(scene_id, "error", "events_parse", f"Failed to parse events.jsonl: {e}"))
    
    # Validate preview.csv
    preview_path = scene_path / "preview.csv"
    if preview_path.exists():
        try:
            with preview_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            stats["preview_row_count"] = len(rows)
            if len(rows) != stats.get("expected_frames", 0):
                errors.append(ValidationError(scene_id, "warning", "preview",
                    f"Preview CSV row count {len(rows)} != expected frames {stats.get('expected_frames', 0)}"))
            
            # Check column presence
            if rows:
                expected_cols = {"tick", "t_ns", "dt_ns", "sleep_drift_ns", "process_ns_estimate", "quality_flags"}
                actual_cols = set(rows[0].keys())
                if expected_cols != actual_cols:
                    missing = expected_cols - actual_cols
                    if missing:
                        errors.append(ValidationError(scene_id, "error", "preview",
                            f"Missing CSV columns: {missing}"))
        except Exception as e:
            errors.append(ValidationError(scene_id, "error", "preview_parse", f"Failed to parse preview.csv: {e}"))
    
    # Validate checksum.txt
    checksum_path = scene_path / "checksum.txt"
    if checksum_path.exists():
        try:
            content = checksum_path.read_text().strip()
            if frames_path.exists():
                actual_sha = hashlib.sha256(frames_path.read_bytes()).hexdigest()
                # Extract checksum from file (format: "sha256  frames.ds64  <hash>")
                parts = content.split()
                if len(parts) >= 3:
                    file_sha = parts[2]
                    if file_sha.lower() != actual_sha.lower():
                        errors.append(ValidationError(scene_id, "error", "checksum",
                            f"SHA256 mismatch: stored={file_sha[:8]}... actual={actual_sha[:8]}..."))
                    else:
                        stats["checksum_verified"] = True
        except Exception as e:
            errors.append(ValidationError(scene_id, "warning", "checksum_parse", f"Failed to validate checksum: {e}"))
    
    valid = len([e for e in errors if e.severity == "error"]) == 0
    return SceneValidationResult(
        scene_id=scene_id,
        label=stats.get("label", "unknown"),
        valid=valid,
        errors=errors,
        stats=stats
    )


def validate_dataset(project_name: str) -> DatasetValidationResult:
    """
    Validate an entire dataset project.
    
    Returns comprehensive validation result including per-scene checks,
    cross-scene comparisons, and quality statistics.
    """
    root = project_path(project_name)
    scenes_dir = root / "scenes"
    
    if not scenes_dir.exists():
        raise FileNotFoundError(f"No scenes directory found in {root}")
    
    scene_paths = sorted(scenes_dir.glob("scene_*"))
    results: list[SceneValidationResult] = []
    
    for scene_path in scene_paths:
        if scene_path.is_dir():
            result = validate_scene(scene_path)
            results.append(result)
    
    # Compute cross-scene comparison statistics
    comparison: dict[str, Any] = {}
    
    if results:
        confidences = [r.stats.get("confidence", 0) for r in results if "confidence" in r.stats]
        if confidences:
            comparison["confidence_stats"] = {
                "min": min(confidences),
                "max": max(confidences),
                "avg": sum(confidences) / len(confidences),
                "range": max(confidences) - min(confidences)
            }
        
        jitters = [r.stats.get("jitter_ns", 0) for r in results if "jitter_ns" in r.stats]
        if jitters:
            comparison["jitter_ns_stats"] = {
                "min": min(jitters),
                "max": max(jitters),
                "avg": sum(jitters) / len(jitters),
                "range": max(jitters) - min(jitters)
            }
        
        labels = [r.label for r in results]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        comparison["labels"] = label_counts
        comparison["unique_labels"] = len(set(labels))
        
        frame_counts = [r.stats.get("actual_frames", 0) for r in results if "actual_frames" in r.stats]
        comparison["total_frames"] = sum(frame_counts)
        comparison["scene_count"] = len(results)
    
    error_count = sum(len([e for e in r.errors if e.severity == "error"]) for r in results)
    warning_count = sum(len([e for e in r.errors if e.severity == "warning"]) for r in results)
    valid_count = sum(1 for r in results if r.valid)
    
    return DatasetValidationResult(
        project_name=project_name,
        total_scenes=len(results),
        valid_scenes=valid_count,
        error_count=error_count,
        warning_count=warning_count,
        scenes=results,
        comparison=comparison
    )


def print_validation_report(result: DatasetValidationResult, verbose: bool = False) -> None:
    """Print a human-readable validation report."""
    print(f"\n{'='*70}")
    print(f"dSense Dataset Validation Report: {result.project_name}")
    print(f"{'='*70}\n")
    
    print(f"Summary:")
    print(f"  Total scenes: {result.total_scenes}")
    print(f"  Valid scenes: {result.valid_scenes}")
    print(f"  Errors: {result.error_count}")
    print(f"  Warnings: {result.warning_count}")
    
    if result.comparison:
        print(f"\nCross-Scene Comparison:")
        if "confidence_stats" in result.comparison:
            stats = result.comparison["confidence_stats"]
            print(f"  Confidence: min={stats['min']:.3f}, max={stats['max']:.3f}, avg={stats['avg']:.3f}, range={stats['range']:.3f}")
        if "jitter_ns_stats" in result.comparison:
            stats = result.comparison["jitter_ns_stats"]
            print(f"  Jitter (ns): min={stats['min']:.0f}, max={stats['max']:.0f}, avg={stats['avg']:.0f}, range={stats['range']:.0f}")
        if "labels" in result.comparison:
            print(f"  Labels: {result.comparison['labels']} ({result.comparison.get('unique_labels', 0)} unique)")
        if "total_frames" in result.comparison:
            print(f"  Total frames: {result.comparison['total_frames']:,}")
    
    print(f"\nPer-Scene Details:")
    print(f"{'  Scene':<20} {'Label':<30} {'Frames':<10} {'Confidence':<12} {'Status':<10}")
    print(f"  {'-'*68}")
    
    for scene_result in result.scenes:
        status = "✓ Valid" if scene_result.valid else "✗ Error"
        frames = scene_result.stats.get("actual_frames", 0)
        conf = scene_result.stats.get("confidence", 0)
        print(f"  {scene_result.scene_id:<20} {scene_result.label:<30} {frames:<10} {conf:<12.3f} {status:<10}")
        
        if verbose and scene_result.errors:
            for error in scene_result.errors:
                severity_symbol = "⚠" if error.severity == "warning" else "✗"
                print(f"    {severity_symbol} [{error.check}] {error.message}")
    
    print(f"\n{'='*70}\n")
