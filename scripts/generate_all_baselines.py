#!/usr/bin/env python
"""
Master script to generate comprehensive baseline datasets for dSense.

This orchestrates the generation of baseline idle and system activity scenarios
to create a comprehensive "system fingerprint." Once you have this baseline,
user presence becomes much more detectable because the system already knows
what it sounds like under various normal conditions.

Usage:
    python scripts/generate_all_baselines.py <project_name> [OPTIONS]

Examples:
    # Generate everything (full baseline)
    python scripts/generate_all_baselines.py demo_lab
    
    # Generate quick baseline (30 min total)
    python scripts/generate_all_baselines.py demo_lab --quick
    
    # Only idle scenarios, dry run to preview
    python scripts/generate_all_baselines.py demo_lab --idle-only --dry-run
    
    # Only activity scenarios
    python scripts/generate_all_baselines.py demo_lab --activity-only
    
    # Custom mix
    python scripts/generate_all_baselines.py demo_lab --idle --no-activity --tick-hz 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scenarios import BASELINE_SCENARIOS, ACTIVITY_SCENARIOS


def run_script(script_name: str, args: list[str]) -> int:
    """Run a generator script as subprocess."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args
    
    try:
        result = subprocess.run(cmd, check=False, cwd=Path(__file__).parent.parent)
        return result.returncode
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return 1


def print_all_scenarios(filter_str: str = None) -> None:
    """Print all available scenarios."""
    print(f"\n{'='*80}")
    print("Idle Baseline Scenarios")
    print(f"{'='*80}\n")
    
    filtered_idle = BASELINE_SCENARIOS
    if filter_str:
        filtered_idle = [s for s in BASELINE_SCENARIOS if filter_str.lower() in s.label.lower()]
    
    for i, scenario in enumerate(filtered_idle, 1):
        print(f"{i:2d}. {scenario.label} ({scenario.duration}s)")
        print(f"    {scenario.description}")
        print()
    
    print(f"Total idle: {len(filtered_idle)} scenarios\n")
    
    print(f"{'='*80}")
    print("System Activity Scenarios")
    print(f"{'='*80}\n")
    
    filtered_activity = ACTIVITY_SCENARIOS
    if filter_str:
        filtered_activity = [s for s in ACTIVITY_SCENARIOS if filter_str.lower() in s.label.lower()]
    
    for i, scenario in enumerate(filtered_activity, 1):
        print(f"{i:2d}. {scenario.label} ({scenario.duration}s)")
        print(f"    {scenario.description}")
        print()
    
    print(f"Total activity: {len(filtered_activity)} scenarios")
    print(f"\n{'='*80}")
    print(f"Grand total: {len(filtered_idle) + len(filtered_activity)} scenarios")
    total_time = (sum(s.duration for s in filtered_idle) + sum(s.duration for s in filtered_activity)) / 60
    print(f"Estimated time: {total_time:.1f} minutes")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="generate_all_baselines.py",
        description="Generate comprehensive baseline datasets for dSense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Baseline Categories:
  idle       - Pure idle and low-activity scenarios (16 scenarios, ~12 min)
  activity   - System load scenarios (12 scenarios, ~9 min)
  all        - Both idle and activity (28 scenarios, ~21 min)

Examples:
  # Full baseline (all scenarios)
  python scripts/generate_all_baselines.py demo_lab
  
  # Quick baseline (just idle)
  python scripts/generate_all_baselines.py demo_lab --idle-only
  
  # Preview without recording
  python scripts/generate_all_baselines.py demo_lab --dry-run
  
  # Higher fidelity at 200 Hz
  python scripts/generate_all_baselines.py demo_lab --tick-hz 200
  
  # Only specific types
  python scripts/generate_all_baselines.py demo_lab --idle-only --filter network
        """
    )
    
    parser.add_argument("project_name", nargs="?", help="Project name (required unless using --list)")
    parser.add_argument("--idle-only", action="store_true", help="Only generate idle scenarios")
    parser.add_argument("--activity-only", action="store_true", help="Only generate activity scenarios")
    parser.add_argument("--quick", action="store_true", help="Quick mode (idle scenarios only)")
    parser.add_argument("--tick-hz", type=int, default=100, help="Sampling rate (default: 100)")
    parser.add_argument("--filter", type=str, help="Only record matching scenarios (case-insensitive)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without recording")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", action="store_true", help="List all scenarios and exit")
    
    args = parser.parse_args()
    
    # Handle --list flag
    if args.list:
        print_all_scenarios(args.filter)
        return 0
    
    # Project name is required unless using --list
    if not args.project_name:
        parser.print_help()
        print("\n✗ Error: project_name is required (unless using --list)")
        return 1
    
    # Determine which scripts to run
    run_idle = True
    run_activity = True
    
    if args.idle_only or args.quick:
        run_activity = False
    elif args.activity_only:
        run_idle = False
    
    print(f"\n{'='*80}")
    print("dSense Comprehensive Baseline Generator")
    print(f"{'='*80}")
    print(f"Project: {args.project_name}")
    print(f"Tick rate: {args.tick_hz} Hz")
    
    to_run = []
    if run_idle:
        to_run.append("idle")
    if run_activity:
        to_run.append("activity")
    
    print(f"Scenarios: {', '.join(to_run)}")
    
    if args.dry_run:
        print("Mode: DRY RUN (preview only)")
    
    print(f"{'='*80}\n")
    
    overall_success = True
    
    # Run idle scenarios
    if run_idle:
        print("\n" + "─"*80)
        print("PHASE 1: Idle Baselines")
        print("─"*80 + "\n")
        
        idle_args = [args.project_name, f"--tick-hz={args.tick_hz}"]
        if args.filter:
            idle_args.append(f"--filter={args.filter}")
        if args.dry_run:
            idle_args.append("--dry-run")
        if args.verbose:
            idle_args.append("--verbose")
        
        if run_script("generate_baseline_dataset.py", idle_args) != 0:
            overall_success = False
    
    # Run activity scenarios
    if run_activity:
        print("\n" + "─"*80)
        print("PHASE 2: System Activity")
        print("─"*80 + "\n")
        
        activity_args = [args.project_name, f"--tick-hz={args.tick_hz}"]
        if args.filter:
            activity_args.append(f"--filter={args.filter}")
        if args.dry_run:
            activity_args.append("--dry-run")
        if args.verbose:
            activity_args.append("--verbose")
        
        if run_script("generate_system_activity_dataset.py", activity_args) != 0:
            overall_success = False
    
    # Final summary
    print(f"\n{'='*80}")
    if overall_success:
        print("✓ Baseline generation complete!")
        print(f"{'='*80}\n")
        print("Next steps:")
        print("  1. Review generated scenes: python -m dsense list-scenes demo_lab")
        print("  2. Validate dataset: python -m dsense validate demo_lab")
        print("  3. Record user presence scenarios with known baseline")
        print()
        return 0
    else:
        print("✗ Some scenarios failed to record")
        print(f"{'='*80}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
