# dSense Autotest Quick Reference

## Overview

The autotest feature performs comprehensive validation on dSense datasets to ensure data integrity, consistency, and quality. Use it after each recording session to catch problems early.

## Basic Usage

### Validate Your Dataset

```bash
python -m dsense validate <project_name>
```

Example:
```bash
python -m dsense validate demo_lab
```

### Verbose Output (Shows Details)

```bash
python -m dsense validate <project_name> --verbose
```

Add `-v` or `--verbose` to see detailed error/warning messages for each scene.

## What Gets Checked

### File Integrity (per scene)
- ✓ All 6 required files exist: `scene.json`, `frames.ds64`, `events.jsonl`, `preview.csv`, `checksum.txt`, `notes.txt`
- ✓ `scene.json` is valid JSON with correct schema
- ✓ `frames.ds64` file size is divisible by 64 bytes (frame size)
- ✓ Frame count matches expected count

### Data Validation
- ✓ SHA-256 checksum verified (frames not corrupted)
- ✓ Each frame has valid BLAKE2s checksum
- ✓ `events.jsonl` contains all required event types
- ✓ Event times are in order and within duration
- ✓ `preview.csv` row count matches frame count
- ✓ CSV has required columns: `tick`, `t_ns`, `dt_ns`, `sleep_drift_ns`, `process_ns_estimate`, `quality_flags`

### Metadata Consistency
- ✓ `scene_id` in `scene.json` matches directory name
- ✓ Frame size declared in JSON matches actual (64 bytes)
- ✓ Duration, tick rate, and expected frames are consistent

### Cross-Scene Analysis
- ✓ Confidence range (min, max, average, variation)
- ✓ Timing jitter statistics
- ✓ Scene label distribution
- ✓ Total frame count
- ✓ Coverage (unique labels and their counts)

## Output Format

### Summary

```
Summary:
  Total scenes: 4
  Valid scenes: 4
  Errors: 0
  Warnings: 0
```

### Cross-Scene Comparison

```
Cross-Scene Comparison:
  Confidence: min=0.997, max=0.997, avg=0.997, range=0.000
  Jitter (ns): min=85105, max=98800, avg=93560, range=13695
  Labels: {'baseline_idle': 1, 'person_walks_front_left_to_right': 3} (2 unique)
  Total frames: 6,000
```

### Per-Scene Details

```
Per-Scene Details:
  Scene              Label                          Frames     Confidence   Status    
  scene_000001       baseline_idle                  3000       0.997        ✓ Valid
  scene_000002       person_walks_front_left_to_right 1000       0.997        ✓ Valid
```

### Errors/Warnings (Verbose Mode)

```
  ✗ [checksum] SHA256 mismatch: stored=abc123... actual=def456...
  ⚠ [preview] Preview CSV row count 950 != expected frames 1000
```

## Interpreting Results

### Green Light ✓ Valid
- All scenes pass validation
- No errors or warnings
- Data is trustworthy for downstream use

### Yellow Light ⚠ Warnings
- Non-critical issues (e.g., empty notes)
- May indicate incomplete metadata
- Scene is still usable but should be reviewed
- Fix: Add notes, check timing windows, etc.

### Red Light ✗ Error
- Critical data integrity issue
- Scene must be reviewed or re-recorded
- Issues: corrupted frames, missing files, checksum failure
- Fix: Check for file corruption, re-record if necessary

## Exit Codes (for scripting)

```python
from dsense.autotest import validate_dataset, print_validation_report

result = validate_dataset("demo_lab")
print_validation_report(result)

# Exit code 0 if all valid, 1 if any errors
exit(0 if result.error_count == 0 else 1)
```

## Common Issues & Solutions

### Issue: "Missing required file: frames.ds64"
**Cause**: Scene recording was interrupted or failed  
**Solution**: Re-record the scene

### Issue: "Frame count mismatch: 950 != 1000"
**Cause**: Some frames dropped during recording or file truncated  
**Solution**: Check system load; re-record with lower tick rate or ensure minimal background activity

### Issue: "SHA256 mismatch"
**Cause**: Frame file corrupted after recording  
**Solution**: Re-record; check disk for errors; check file system permissions

### Issue: "Preview CSV row count != expected frames"
**Cause**: Preview export missed some frames (rare)  
**Solution**: Re-run `python -m dsense export-preview` to regenerate

### Issue: "Confidence < 0.95"
**Cause**: Timing jitter too high (unstable system)  
**Solution**: Close background apps, disable power management features, reduce tick rate

## Workflow: Validate After Each Session

```bash
# 1. Record scenes
python -m dsense scene demo_lab --label "my_scenario" --repeat 2

# 2. Validate immediately
python -m dsense validate demo_lab

# 3. If errors, fix and re-record:
# - If file-related: re-record scene
# - If metadata-related: edit notes, check timing

# 4. If all ✓: you're good to continue
```

## Saving Validation Reports

```bash
# Timestamp-based archive
python -m dsense validate demo_lab > validation_$(date +%Y%m%d_%H%M%S).txt

# Compare reports over time
diff validation_20260627.txt validation_20260628.txt
```

## Python API (Advanced)

```python
from dsense.autotest import validate_dataset, validate_scene, print_validation_report

# Validate entire project
result = validate_dataset("demo_lab")

# Check results
if result.error_count == 0:
    print("✓ All scenes valid!")
else:
    print(f"✗ Found {result.error_count} errors")

# Inspect individual scenes
for scene_result in result.scenes:
    print(f"{scene_result.scene_id}: confidence={scene_result.stats['confidence']}")
    for error in scene_result.errors:
        print(f"  - {error.severity}: {error.message}")

# Print report
print_validation_report(result, verbose=True)
```

## Checklist: Good Dataset Indicators

- [ ] `Total scenes ≥ 3`
- [ ] `Valid scenes == Total scenes` (no errors)
- [ ] `Errors == 0`
- [ ] `Warnings < 3` (or none if possible)
- [ ] `Unique labels ≥ 2` (variety)
- [ ] `Average confidence ≥ 0.99`
- [ ] `Jitter range < 50,000 ns` (timing stable)

## See Also

- [DATASET_REVIEW.md](DATASET_REVIEW.md) — Full dataset assessment and recommendations
- [docs/dataset-format.md](docs/dataset-format.md) — Scene file format specification
- [dsense/autotest.py](dsense/autotest.py) — Implementation details and docstrings
