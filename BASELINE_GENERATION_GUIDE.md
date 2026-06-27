# Baseline Dataset Generation Guide

This guide explains how to use the baseline generation scripts to create a comprehensive "system fingerprint" for dSense. With a solid baseline, user presence detection becomes much clearer because the system already knows what normal looks like under various conditions.

## Why Baselines Matter

### The Problem
- Without baselines, it's hard to distinguish between normal system noise and actual events
- Every system sounds different (thermal patterns, scheduler behavior, I/O characteristics)
- Background processes create noise that looks like user presence

### The Solution
- Capture "what the system sounds like" in various idle/low-activity states
- Record system behavior under different loads (CPU, I/O, network, thermal)
- This gives you a comprehensive reference model
- User presence becomes detectable as *deviation from baseline*

## Available Scripts

### 1. `generate_baseline_dataset.py`
Records pure idle and low-activity scenarios (16 scenarios, ~12 minutes total)

**Scenarios included:**
- Early morning, daytime, evening idle
- Idle with fan, network, disk activity
- Low activity with text editor, static browser page
- Different power states (performance vs power-saver)
- Environmental variations (quiet vs ambient noise)
- Post-boot and long-idle conditions
- Battery vs plugged-in states
- External display connected

### 2. `generate_system_activity_dataset.py`
Records controlled system load scenarios (12 scenarios, ~9 minutes total)

**Scenarios included:**
- Light, moderate, and heavy CPU load
- Memory pressure (swap activity)
- Disk reads, writes, and random I/O
- Network download and upload
- Process spawning
- Context switch heavy
- Interrupt-driven loads

### 3. `generate_all_baselines.py`
Master orchestrator that runs idle + activity scenarios (28 scenarios, ~21 minutes total)

## Quick Start

### Option 1: Generate Complete Baseline (Recommended)
```bash
cd /home/epi13/dSense

# Full comprehensive baseline (all 28 scenarios, ~21 min)
python scripts/generate_all_baselines.py demo_lab

# Or with higher fidelity at 200 Hz
python scripts/generate_all_baselines.py demo_lab --tick-hz 200
```

### Option 2: Quick Baseline (Just Idle)
```bash
# Fast baseline - only idle scenarios (~12 min)
python scripts/generate_all_baselines.py demo_lab --quick

# Or directly use idle-only script
python scripts/generate_baseline_dataset.py demo_lab
```

### Option 3: Preview First (Dry Run)
```bash
# See what would be recorded without actually recording
python scripts/generate_all_baselines.py demo_lab --dry-run

# List all available scenarios
python scripts/generate_all_baselines.py demo_lab --list
```

## Common Usage Patterns

### Full Baseline Generation
```bash
python scripts/generate_all_baselines.py demo_lab
```
- Records all 28 scenarios
- Total time: ~21 minutes
- Creates comprehensive system fingerprint
- Validates results automatically

### Filter to Specific Scenarios
```bash
# Only CPU-related scenarios
python scripts/generate_all_baselines.py demo_lab --filter cpu

# Only network scenarios
python scripts/generate_baseline_dataset.py demo_lab --filter network

# Only power-related baselines
python scripts/generate_baseline_dataset.py demo_lab --filter power
```

### Higher Fidelity Capture (200 Hz)
```bash
python scripts/generate_all_baselines.py demo_lab --tick-hz 200
```
- Captures at 200 Hz instead of default 100 Hz
- Better temporal resolution
- Takes 2x longer
- Useful for detailed substrate analysis

### Separate Phases
```bash
# Phase 1: Get idle baselines first
python scripts/generate_baseline_dataset.py demo_lab

# Later, add activity baselines
python scripts/generate_system_activity_dataset.py demo_lab

# Or just activity scenarios
python scripts/generate_system_activity_dataset.py demo_lab --filter cpu
```

### Incremental Development
```bash
# Test with just early morning scenario
python scripts/generate_baseline_dataset.py demo_lab --filter "early_morning" --dry-run

# Actual recording
python scripts/generate_baseline_dataset.py demo_lab --filter "early_morning"

# Validate immediately
python -m dsense validate demo_lab --verbose

# Add more as needed
python scripts/generate_baseline_dataset.py demo_lab --filter "daytime"
```

## Output & Validation

Each script automatically:
1. **Records scenarios** with proper metadata
2. **Validates each scene** as it's recorded
3. **Generates a final report** with:
   - Number of successful/failed recordings
   - Total frames captured
   - Validation results
   - Quality statistics

### Check Progress
```bash
# See all scenes recorded so far
python -m dsense list-scenes demo_lab

# Full validation report
python -m dsense validate demo_lab

# Detailed validation with errors/warnings
python -m dsense validate demo_lab --verbose

# Export for analysis
python -m dsense export-preview demo_lab
```

## Understanding the Output

### Recording Summary
```
✓ Successfully recorded: 28 scenes
  Total frames: 280,000
  Time elapsed: 21.3 minutes
  Average frame rate: 218 fps
```

### Validation Report
```
Summary:
  Total scenes: 28
  Valid scenes: 28
  Errors: 0
  Warnings: 0

Cross-Scene Comparison:
  Confidence: min=0.995, max=0.998, avg=0.996, range=0.003
  Jitter (ns): min=45000, max=120000, avg=78500, range=75000
  Labels: {many unique labels} (26 unique)
  Total frames: 280,000
```

## Troubleshooting

### Issue: Script seems stuck
**Cause:** Normal - baseline generation takes time (up to 30 min for full set)  
**Solution:** Be patient, monitor with `ps aux | grep dsense`

### Issue: Some scenarios fail to record
**Cause:** System resource constraints or I/O issues  
**Solution:** 
```bash
# Try re-running with verbose output
python scripts/generate_baseline_dataset.py demo_lab --verbose

# Run only failed scenarios
python scripts/generate_baseline_dataset.py demo_lab --filter "cpu"
```

### Issue: Confidence score below 0.99
**Cause:** System jitter or heavy background load  
**Solution:**
- Close unnecessary applications
- Disable power management features
- Run at different time when system is quieter
- Try again with `--tick-hz 50` for lower frequency

### Issue: "Frame count mismatch"
**Cause:** Some frames dropped during recording  
**Solution:**
- Reduce system load
- Lower tick rate: `--tick-hz 50`
- Re-run validation: `python -m dsense validate demo_lab`

## Best Practices

### 1. Record in Consistent Environment
- Same room, same time of day ideally
- Consistent ambient conditions
- Minimal background activity

### 2. Start with Quick Baseline
```bash
# First: Fast 12-minute idle baseline
python scripts/generate_baseline_dataset.py demo_lab

# Then: Validate
python -m dsense validate demo_lab

# Then: Add activity scenarios if time allows
python scripts/generate_system_activity_dataset.py demo_lab
```

### 3. Document Your Setup
Add notes about your environment in scene metadata:
```bash
# When manually recording, include detailed notes
python -m dsense scene demo_lab \
  --label custom_scenario \
  --notes "Room temp 22C, ambient AC noise, no external load"
```

### 4. Archive and Compare
```bash
# Save validation report
python -m dsense validate demo_lab > baseline_validation_$(date +%Y%m%d_%H%M%S).txt

# Track changes over time
diff baseline_validation_20260626.txt baseline_validation_20260627.txt
```

### 5. Monitor Quality Metrics
```bash
# After each phase, check confidence
python -m dsense validate demo_lab | grep -i confidence

# All scenes should be ≥0.99
# If lower, investigate system conditions
```

## Example Workflow

### Day 1: Quick Baseline
```bash
# 15 minutes of setup and recording
python scripts/generate_baseline_dataset.py demo_lab --verbose

# Validate
python -m dsense validate demo_lab
# Output: 16 scenes, all valid, 0.996 avg confidence ✓
```

### Day 2: Activity Scenarios
```bash
# Add system load baselines
python scripts/generate_system_activity_dataset.py demo_lab

# Final validation
python -m dsense validate demo_lab --verbose
# Output: 28 scenes total, all valid ✓
```

### Day 3: User Presence Scenarios
```bash
# Now that baseline is solid, record user presence
python -m dsense scene demo_lab \
  --label "user_walks_in" \
  --pre-roll 2 --action 3 --post-roll 2 \
  --repeat 3

# Validate
python -m dsense validate demo_lab
# Observe how user presence differs from baseline!
```

## Advanced: Customizing Scenarios

To add your own scenarios, edit the script:

```python
# In generate_baseline_dataset.py or generate_system_activity_dataset.py
BASELINE_SCENARIOS = [
    # ... existing scenarios ...
    
    BaselineScenario(
        label="my_custom_scenario",
        duration=30,
        description="My specific condition",
        notes="Details about what makes this unique",
    ),
]
```

Then re-run:
```bash
python scripts/generate_baseline_dataset.py demo_lab --filter "custom"
```

## Summary

| Task | Command | Time |
|------|---------|------|
| Quick baseline (idle only) | `python scripts/generate_all_baselines.py demo_lab --quick` | 12 min |
| Full comprehensive baseline | `python scripts/generate_all_baselines.py demo_lab` | 21 min |
| Activity scenarios only | `python scripts/generate_system_activity_dataset.py demo_lab` | 9 min |
| Preview without recording | `python scripts/generate_all_baselines.py demo_lab --dry-run` | 1 min |
| Validate dataset | `python -m dsense validate demo_lab` | <1 min |
| List all scenarios | `python scripts/generate_all_baselines.py demo_lab --list` | <1 min |

---

**Next Step:** Once you have a solid baseline, use it to detect user presence! The substrate data will show clear patterns when humans interact with your system.
