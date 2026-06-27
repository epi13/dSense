# Baseline Dataset Generation Scripts

This directory contains scripts to automatically generate comprehensive baseline system fingerprints for dSense. These baselines capture "what your system sounds like" under various idle and loaded conditions, making user presence detection much clearer.

## Quick Start

### Generate Everything (Recommended)
```bash
cd /home/epi13/dSense
python scripts/generate_all_baselines.py demo_lab
```

This records **28 scenarios** (~21 minutes):
- 17 idle/low-activity baselines
- 12 system load baselines

### Generate Just Idle (Quick)
```bash
python scripts/generate_baseline_dataset.py demo_lab
```

Records **17 idle scenarios** (~12 minutes total).

### Preview Without Recording (Dry Run)
```bash
python scripts/generate_all_baselines.py demo_lab --dry-run
```

### List All Scenarios
```bash
python scripts/generate_all_baselines.py --list
```

## Scripts Overview

### 1. `generate_all_baselines.py`
Master orchestrator that runs idle + activity generation in sequence.

**Features:**
- Runs both idle and activity generators
- Configurable combination of baselines
- Automatic dataset validation
- Progress tracking

**Usage:**
```bash
# Full baseline
python scripts/generate_all_baselines.py demo_lab

# Idle only (quick)
python scripts/generate_all_baselines.py demo_lab --quick

# Only activity scenarios
python scripts/generate_all_baselines.py demo_lab --activity-only

# With filter (both idle and activity)
python scripts/generate_all_baselines.py demo_lab --filter cpu

# Higher fidelity (200 Hz)
python scripts/generate_all_baselines.py demo_lab --tick-hz 200

# Preview without recording
python scripts/generate_all_baselines.py demo_lab --dry-run

# List all scenarios
python scripts/generate_all_baselines.py --list
```

### 2. `generate_baseline_dataset.py`
Records idle and low-activity baseline scenarios.

**Scenarios (17 total, ~12 minutes):**
- Early morning, daytime, evening idle
- Idle with fan, network, or disk activity
- Low activity with applications (editor, browser)
- High performance vs power saver mode
- Quiet vs ambient noise environments
- After boot vs after long idle
- Battery low vs plugged in
- External display connected

**Usage:**
```bash
# Generate all
python scripts/generate_baseline_dataset.py demo_lab

# Filter to specific scenarios
python scripts/generate_baseline_dataset.py demo_lab --filter "daytime"
python scripts/generate_baseline_dataset.py demo_lab --filter "fan"
python scripts/generate_baseline_dataset.py demo_lab --filter "power"

# Dry run preview
python scripts/generate_baseline_dataset.py demo_lab --dry-run

# Verbose output
python scripts/generate_baseline_dataset.py demo_lab --verbose

# List scenarios
python scripts/generate_baseline_dataset.py --list

# Higher tick rate
python scripts/generate_baseline_dataset.py demo_lab --tick-hz 200
```

### 3. `generate_system_activity_dataset.py`
Records controlled system load scenarios.

**Scenarios (12 total, ~9 minutes):**
- CPU: light (20%), moderate (50%), heavy (90%+)
- Memory pressure (swap activity)
- I/O: reads, writes, random access
- Network: download and upload
- Process spawning
- Context switch heavy
- Interrupt-driven loads

**Usage:**
```bash
# Generate all
python scripts/generate_system_activity_dataset.py demo_lab

# Filter to activity type
python scripts/generate_system_activity_dataset.py demo_lab --filter cpu
python scripts/generate_system_activity_dataset.py demo_lab --filter io
python scripts/generate_system_activity_dataset.py demo_lab --filter network

# Preview without recording
python scripts/generate_system_activity_dataset.py demo_lab --dry-run

# List scenarios
python scripts/generate_system_activity_dataset.py --list
```

### 4. `scenarios.py`
Shared scenario definitions used by all generator scripts.

**Contents:**
- `BASELINE_SCENARIOS`: List of idle scenarios
- `ACTIVITY_SCENARIOS`: List of system load scenarios
- `BaselineScenario` dataclass: Scenario configuration

## Common Workflows

### Workflow 1: Full Baseline (Recommended)
```bash
# Step 1: Generate everything
python scripts/generate_all_baselines.py demo_lab

# Step 2: Check progress
python -m dsense list-scenes demo_lab

# Step 3: Validate
python -m dsense validate demo_lab --verbose

# Step 4: Export for analysis
python -m dsense export-preview demo_lab
```

### Workflow 2: Incremental Build
```bash
# Day 1: Quick baseline (idle only)
python scripts/generate_baseline_dataset.py demo_lab
python -m dsense validate demo_lab

# Day 2: Add activity scenarios
python scripts/generate_system_activity_dataset.py demo_lab
python -m dsense validate demo_lab

# Day 3: Start recording user presence
python -m dsense scene demo_lab --label "user_walks_in" --repeat 3
python -m dsense validate demo_lab
```

### Workflow 3: Specific Scenario Capture
```bash
# Capture only CPU-related baselines
python scripts/generate_system_activity_dataset.py demo_lab --filter cpu

# Or only idle baselines
python scripts/generate_baseline_dataset.py demo_lab --filter "idle"

# Validate just those
python -m dsense validate demo_lab
```

### Workflow 4: High-Fidelity Capture
```bash
# For detailed substrate analysis, use 200 Hz tick rate
python scripts/generate_all_baselines.py demo_lab --tick-hz 200

# Takes 2x longer but better temporal resolution
```

## Output

Each script generates:

1. **Recorded Scenes**
   - One scene directory per baseline scenario
   - All required files: `scene.json`, `frames.ds64`, `events.jsonl`, `preview.csv`, `checksum.txt`, `notes.txt`

2. **Progress Output**
   - Real-time recording status
   - Frame count per scenario
   - Total time elapsed
   - FPS achieved

3. **Validation Report** (automatic after recording)
   - Scene validity check
   - Cross-scene quality comparison
   - Error/warning summary

### Example Output
```
================================================================================
dSense Baseline Dataset Generator
================================================================================
Project: demo_lab
Scenarios to record: 17
Total estimated time: 12.2 minutes
Tick rate: 100 Hz

────────────────────────────────────────────────────────────────────────────────

[ 1/17] baseline_idle_early_morning
         Recording to scene_000005... ✓ 6,000 frames

[ 2/17] baseline_idle_daytime
         Recording to scene_000006... ✓ 6,000 frames

...

================================================================================
Recording Summary
================================================================================

✓ Successfully recorded: 17 scenes
  Total frames: 102,000
  Time elapsed: 12.1 minutes
  Average frame rate: 140 fps

================================================================================
```

## Command Reference

### Master Script
```bash
python scripts/generate_all_baselines.py [PROJECT] [OPTIONS]

Options:
  --idle-only        Only generate idle scenarios
  --activity-only    Only generate activity scenarios
  --quick            Quick mode (idle only)
  --tick-hz HZ       Sampling rate (default: 100)
  --filter FILTER    Only record matching scenarios
  --dry-run          Preview without recording
  --verbose, -v      Detailed output
  --list             List all scenarios and exit
```

### Idle Generator
```bash
python scripts/generate_baseline_dataset.py [PROJECT] [OPTIONS]

Options:
  --tick-hz HZ       Sampling rate (default: 100)
  --filter FILTER    Only record matching scenarios
  --dry-run          Preview without recording
  --verbose, -v      Detailed output
  --list             List all scenarios and exit
```

### Activity Generator
```bash
python scripts/generate_system_activity_dataset.py [PROJECT] [OPTIONS]

Options:
  --tick-hz HZ       Sampling rate (default: 100)
  --filter FILTER    Only record matching scenarios
  --dry-run          Preview without recording
  --verbose, -v      Detailed output
  --list             List all scenarios and exit
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'dsense'
**Cause:** Python path not set correctly  
**Solution:** Set PYTHONPATH before running:
```bash
export PYTHONPATH=/home/epi13/dSense
python scripts/generate_baseline_dataset.py demo_lab
```

Or run from project root:
```bash
cd /home/epi13/dSense
python scripts/generate_baseline_dataset.py demo_lab
```

### Issue: Scripts seem to hang
**Cause:** Normal - baseline generation takes time (up to 25 min for full set)  
**Solution:** Be patient, monitor progress in terminal. Use `--dry-run` to preview first.

### Issue: Some scenarios fail to record
**Cause:** System resource constraints  
**Solution:**
- Close unnecessary applications
- Try running just failed scenarios with `--filter`
- Re-run full script to retry

### Issue: Confidence scores low (<0.99)
**Cause:** System too busy or jittery  
**Solution:**
- Run during quiet time
- Close background applications
- Try lower tick rate: `--tick-hz 50`

## Best Practices

1. **Start with dry-run** to see what will be recorded
2. **Use quick mode first** (`--quick`) to get 80% coverage faster
3. **Monitor disk space** - full baseline can be 10-20 MB
4. **Validate after each phase** to catch issues early
5. **Use verbose mode** (`--verbose`) for troubleshooting
6. **Archive results** - save validation reports for comparison over time

## Integration with CI/CD

Add baseline generation to your workflow:

```bash
#!/bin/bash
set -e

cd /path/to/dSense

# Generate baseline
python scripts/generate_all_baselines.py test_project

# Validate
python -m dsense validate test_project --verbose

# Export for analysis
python -m dsense export-preview test_project

# Archive report
python -m dsense validate test_project > reports/baseline_$(date +%Y%m%d).txt
```

## Advanced: Customizing Scenarios

To add your own baseline scenario, edit `scenarios.py`:

```python
from scenarios import BaselineScenario

# Add to BASELINE_SCENARIOS list:
BaselineScenario(
    label="my_custom_scenario",
    duration=30,
    description="My specific condition",
    notes="Details about what makes this special",
)
```

Then generate with filter:
```bash
python scripts/generate_baseline_dataset.py demo_lab --filter "custom"
```

## See Also

- [BASELINE_GENERATION_GUIDE.md](../BASELINE_GENERATION_GUIDE.md) — Detailed usage guide
- [AUTOTEST_GUIDE.md](../AUTOTEST_GUIDE.md) — Dataset validation guide
- [dsense/autotest.py](../dsense/autotest.py) — Validation implementation

---

**With a solid baseline, user presence detection becomes clear!** 🎯
