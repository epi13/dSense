# dSense Dataset Review & Recommendations

## ✅ Current State Summary

Your first dataset capture is **excellent**:

| Metric | Result |
|--------|--------|
| Total Scenes | 4 |
| Valid Scenes | 4 |
| All Checksums | ✓ Valid |
| Confidence (all) | 0.997 |
| Frame Integrity | ✓ Perfect |
| Total Frames Captured | 6,000 |

### Scene Breakdown
- **scene_000001** (30s): `baseline_idle` — excellent idle baseline (0.997 confidence)
- **scene_000002** (10s): `person_walks_front_left_to_right` (0.997 confidence)
- **scene_000003** (10s): `person_walks_front_left_to_right` (0.997 confidence)  
- **scene_000004** (10s): `person_walks_front_left_to_right` (0.997 confidence)

### Quality Metrics
- **Jitter Range**: 85–99 µs (very clean, ~13 µs variation)
- **Frame Accuracy**: All scenes captured exact frame counts (0 dropped/late)
- **Data Integrity**: All SHA-256 checksums verified

---

## 🎯 Recommendations & Next Steps

### 1. **Expand Scene Variety** (High Priority)

Currently, 75% of scenes are the same label (`person_walks_front_left_to_right`). For robust training data, capture:

```bash
# More baseline scenarios
python -m dsense record-baseline demo_lab --duration 60 --notes "idle with background music"
python -m dsense record-baseline demo_lab --duration 30 --notes "idle with fan running"

# Environmental variations
python -m dsense scene demo_lab \
  --label "cpu_load_high" \
  --duration 10 --pre-roll 2 --action 5 --post-roll 3 \
  --notes "CPU loop running during action window"

python -m dsense scene demo_lab \
  --label "network_activity" \
  --duration 10 --pre-roll 2 --action 5 --post-roll 3 \
  --notes "Large file downloads during action"

python -m dsense scene demo_lab \
  --label "keyboard_mouse_input" \
  --duration 10 --pre-roll 2 --action 5 --post-roll 3 \
  --notes "Active typing and clicking"

python -m dsense scene demo_lab \
  --label "thermal_spike" \
  --duration 15 --pre-roll 3 --action 7 --post-roll 5 \
  --notes "System under sustained load"

# More spatial variations
python -m dsense scene demo_lab \
  --label "person_walks_right_to_left" \
  --duration 10 --pre-roll 2 --action 5 --post-roll 3 \
  --repeat 2 --notes "opposite walkby direction"

python -m dsense scene demo_lab \
  --label "person_stationary_nearby" \
  --duration 10 --pre-roll 2 --action 5 --post-roll 3 \
  --repeat 2 --notes "person standing still near machine"
```

### 2. **Add Scene Descriptions** (Medium Priority)

Enhance each scene's `notes.txt` with:
- Physical environment (room noise, temperature, lighting)
- Machine state (CPU usage, network, power profile)
- Specific actions during action window
- Any anomalies or concerns

Example:
```
Room: quiet lab, ambient temperature 22°C
Machine: MacBook Pro, ~15% CPU baseline, no external load
Action: Person walks slowly from left side to right side, 5 feet away
Notes: Background AC hum present, consistent pace
```

### 3. **Use Autotest After Each Capture Session** (Critical Workflow)

After recording scenes:

```bash
# Quick check
python -m dsense validate demo_lab

# Detailed check with verbose output
python -m dsense validate demo_lab --verbose

# Store result for comparison
python -m dsense validate demo_lab > validation_$(date +%Y%m%d_%H%M%S).txt
```

This validates:
- ✓ All required files exist
- ✓ JSON schema integrity
- ✓ Frame file format and size
- ✓ SHA-256 checksums
- ✓ CSV alignment
- ✓ Event timing consistency
- ✓ Cross-scene quality comparison

### 4. **Create Representative Baseline Dataset**

Establish a "golden baseline" for your environment:

```bash
# Clear demo_lab and start fresh with diverse captures
rm -rf datasets/demo_lab
python -m dsense init demo_lab

# Capture 5–10 different scenarios, 2–3 repeats each
# Then validate
python -m dsense validate demo_lab

# Export for downstream analysis
python -m dsense export-preview demo_lab
```

### 5. **Set Up Continuous Quality Checks** (Optional Automation)

Add to your CI/CD or pre-commit:

```python
# scripts/validate_dataset.py
#!/usr/bin/env python
from dsense.autotest import validate_dataset, print_validation_report

result = validate_dataset("demo_lab")
print_validation_report(result, verbose=True)

if result.error_count > 0 or result.valid_scenes < result.total_scenes:
    exit(1)
```

---

## 📊 Understanding the Validation Report

### Summary Section
- **Total scenes**: Count of all scenes found
- **Valid scenes**: Scenes with no critical errors
- **Errors**: Critical issues (missing files, corrupted data)
- **Warnings**: Non-critical issues (missing notes, rare outliers)

### Cross-Scene Comparison
- **Confidence**: Min/max/avg/range of model confidence across all scenes
- **Jitter**: Timing variation (lower = more stable)
- **Labels**: Count of each scene label (look for variety!)
- **Total frames**: Aggregate frame count

### Per-Scene Details
- **Scene ID**: Unique identifier
- **Label**: Scene type/category
- **Frames**: Number of 64-byte frames captured
- **Confidence**: Quality metric (>0.95 is excellent)
- **Status**: ✓ Valid or ✗ Error

---

## 🔍 Checklist for Dataset Readiness

- [ ] **Variety**: At least 6 unique scene labels
- [ ] **Coverage**: 2–3 captures per label (for robustness)
- [ ] **Documentation**: All scenes have detailed notes
- [ ] **Validation**: `validate demo_lab` passes with 0 errors
- [ ] **Baseline**: At least 3 different baseline scenarios
- [ ] **Quality**: Confidence ≥0.99 for all scenes
- [ ] **Metadata**: All machine_state, pre-roll, action, post-roll timings accurate

---

## 🚀 Next Session Quick Commands

```bash
# Start work
cd /home/epi13/dSense

# Record a new scene
python -m dsense scene demo_lab --label "your_scenario" --notes "description"

# Check progress
python -m dsense list-scenes demo_lab

# Validate dataset
python -m dsense validate demo_lab

# Export and review
python -m dsense export-preview demo_lab
```

---

## 📚 API Reference: Autotest Functions

See [dsense/autotest.py](../dsense/autotest.py) for:
- `validate_scene(scene_path)` — Validate single scene
- `validate_dataset(project_name)` — Validate entire project
- `ValidationError` — Error/warning dataclass
- `SceneValidationResult` — Result for one scene
- `DatasetValidationResult` — Result for full dataset
- `print_validation_report(result, verbose=False)` — Pretty-print results

---

## 🤔 FAQ

**Q: Why is confidence 0.997 vs 1.0?**  
A: Tiny jitter (~100ns) is normal in real systems. 0.99+ is excellent; >0.95 is acceptable.

**Q: What if I see warnings?**  
A: Warnings aren't critical but indicate something to review. Errors must be fixed.

**Q: Can I edit scene.json manually?**  
A: Technically yes, but re-running validation after edits is essential.

**Q: How many frames do I need?**  
A: More is better for ML. Aim for 1000+ frames per label (10s at 100 Hz).

---

## 📝 Summary

Your first session captured **perfect-quality data** with excellent timing consistency and data integrity. The autotest feature ensures every future capture stays trustworthy. Focus next on **scenario variety** and **detailed documentation**, then you'll have a solid foundation for substrate-aware AI training.

Good work! 🎉
