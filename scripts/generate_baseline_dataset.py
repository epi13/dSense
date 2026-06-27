#!/usr/bin/env python
"""
Auto-generate baseline system scenes for dSense.

Records multiple idle/low-activity scenarios to build a comprehensive
"system fingerprint" baseline. This makes user presence detection easier
because the system already knows what normal looks like under various conditions.

Usage:
    python scripts/generate_baseline_dataset.py <project_name> [--tick-hz 100] [--filter FILTER]

Examples:
    python scripts/generate_baseline_dataset.py demo_lab
    python scripts/generate_baseline_dataset.py demo_lab --tick-hz 200
    python scripts/generate_baseline_dataset.py demo_lab --filter "idle"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dsense.manifest import init_project, allocate_scene_id, project_path
from dsense.recorder import record_scene
from dsense.autotest import validate_dataset, print_validation_report
from scenarios import BASELINE_SCENARIOS


def print_scenarios(filter_str: str = None) -> None:
    """Print all available baseline scenarios."""
    print(f"\n{'='*80}")
    print("Available Baseline Scenarios")
    print(f"{'='*80}\n")
    
    filtered = BASELINE_SCENARIOS
    if filter_str:
        filtered = [s for s in BASELINE_SCENARIOS if filter_str.lower() in s.label.lower()]
    
    for i, scenario in enumerate(filtered, 1):
        print(f"{i:2d}. {scenario.label} ({scenario.duration}s)")
        print(f"    {scenario.description}")
        print(f"    Notes: {scenario.notes}")
        print()
    
    print(f"Total: {len(filtered)} scenarios")
    print(f"{'='*80}\n")


def record_baseline_dataset(project_name: str, tick_hz: int = 100, filter_str: str = None, 
                           dry_run: bool = False, verbose: bool = False) -> dict:
    """
    Record all baseline scenarios for a project.
    
    Args:
        project_name: Name of the project
        tick_hz: Sampling rate in Hz
        filter_str: Only record scenarios matching this string
        dry_run: Show what would be recorded without actually recording
        verbose: Print detailed information
    
    Returns:
        Dictionary with recording statistics
    """
    init_project(project_name)
    
    # Filter scenarios if requested
    scenarios = BASELINE_SCENARIOS
    if filter_str:
        scenarios = [s for s in BASELINE_SCENARIOS if filter_str.lower() in s.label.lower()]
    
    if not scenarios:
        print(f"✗ No scenarios match filter: {filter_str}")
        return {}
    
    print(f"\n{'='*80}")
    print(f"dSense Baseline Dataset Generator")
    print(f"{'='*80}")
    print(f"Project: {project_name}")
    print(f"Scenarios to record: {len(scenarios)}")
    print(f"Total estimated time: {sum(s.duration for s in scenarios) / 60:.1f} minutes")
    print(f"Tick rate: {tick_hz} Hz")
    
    if dry_run:
        print("\n⚠️  DRY RUN - No scenes will be recorded\n")
    else:
        print(f"\n{'─'*80}\n")
    
    recorded_scenes = []
    failed_scenes = []
    start_time = time.time()
    
    for idx, scenario in enumerate(scenarios, 1):
        print(f"[{idx:2d}/{len(scenarios)}] {scenario.label}")
        
        if verbose:
            print(f"         Duration: {scenario.duration}s")
            print(f"         {scenario.description}")
            print(f"         Notes: {scenario.notes}")
        
        if dry_run:
            print("         [DRY RUN] Skipped\n")
            continue
        
        try:
            scene_id = allocate_scene_id(project_name)
            scene_dir = project_path(project_name) / "scenes" / scene_id
            
            print(f"         Recording to {scene_id}...", end=" ", flush=True)
            
            record_scene(
                scene_dir,
                scene_id,
                scenario.label,
                duration=scenario.duration,
                tick_hz=tick_hz,
                pre_roll=0,
                action=scenario.duration,
                post_roll=0,
                notes=scenario.notes,
                mode="record"
            )
            
            frames = scene_dir / "frames.ds64"
            frame_count = frames.stat().st_size // 64
            
            print(f"✓ {frame_count:,} frames")
            recorded_scenes.append((scenario.label, scene_id, frame_count))
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed_scenes.append((scenario.label, str(e)))
    
    elapsed = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print("Recording Summary")
    print(f"{'='*80}\n")
    
    if recorded_scenes:
        print(f"✓ Successfully recorded: {len(recorded_scenes)} scenes")
        total_frames = sum(count for _, _, count in recorded_scenes)
        print(f"  Total frames: {total_frames:,}")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes")
        print(f"  Average frame rate: {total_frames / elapsed:.0f} fps")
    
    if failed_scenes:
        print(f"\n✗ Failed: {len(failed_scenes)} scenes")
        for label, error in failed_scenes:
            print(f"  - {label}: {error}")
    
    # Validate the recorded dataset
    if recorded_scenes and not dry_run:
        print(f"\n{'─'*80}\n")
        print("Running dataset validation...\n")
        
        try:
            result = validate_dataset(project_name)
            print_validation_report(result, verbose=False)
            
            stats = {
                "total_recorded": len(recorded_scenes),
                "total_failed": len(failed_scenes),
                "total_frames": total_frames,
                "elapsed_seconds": elapsed,
                "validation_valid_scenes": result.valid_scenes,
                "validation_total_scenes": result.total_scenes,
                "validation_errors": result.error_count,
                "validation_warnings": result.warning_count,
            }
        except Exception as e:
            print(f"⚠️  Validation failed: {e}\n")
            stats = {
                "total_recorded": len(recorded_scenes),
                "total_failed": len(failed_scenes),
                "total_frames": total_frames,
                "elapsed_seconds": elapsed,
            }
    else:
        stats = {
            "total_recorded": len(recorded_scenes),
            "total_failed": len(failed_scenes),
            "elapsed_seconds": elapsed,
        }
    
    print(f"\n{'='*80}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        prog="generate_baseline_dataset.py",
        description="Auto-generate baseline system scenes for dSense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all baseline scenarios
  python scripts/generate_baseline_dataset.py demo_lab
  
  # Generate only idle-related scenarios
  python scripts/generate_baseline_dataset.py demo_lab --filter idle
  
  # Preview without recording
  python scripts/generate_baseline_dataset.py demo_lab --dry-run
  
  # Higher tick rate
  python scripts/generate_baseline_dataset.py demo_lab --tick-hz 200
  
  # List available scenarios
  python scripts/generate_baseline_dataset.py --list
        """
    )
    
    parser.add_argument("project_name", nargs="?", help="Project name (required unless using --list)")
    parser.add_argument("--tick-hz", type=int, default=100, help="Sampling rate in Hz (default: 100)")
    parser.add_argument("--filter", type=str, help="Only record scenarios matching this string (case-insensitive)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be recorded without actually recording")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
    parser.add_argument("--list", action="store_true", help="List all available scenarios and exit")
    
    args = parser.parse_args()
    
    # Handle --list flag
    if args.list:
        print_scenarios(args.filter)
        return 0
    
    # Project name is required unless using --list
    if not args.project_name:
        parser.print_help()
        print("\n✗ Error: project_name is required (unless using --list)")
        return 1
    
    # Record the baseline dataset
    try:
        record_baseline_dataset(
            args.project_name,
            tick_hz=args.tick_hz,
            filter_str=args.filter,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        return 0
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
