# dSense

dSense gives software a body by turning the physical behavior of its host machine into a continuous sensory stream.

## What dSense is

dSense, originally “Digital Senses,” is a machine-proprioception project. It treats a computer as a small embedded physical universe rather than a perfectly clean abstract simulator. Real machines have timing jitter, scheduler interference, cache and IO latency, thermal drift, power-state transitions, fan cycles, network delay, and other substrate behavior. dSense exposes those intrinsic signals as sensory channels that future adaptive systems and AI models can learn from.

The long-term goal is substrate-aware AI: software that can synchronize with, adapt to, and cautiously reason about the physical and operating environment beneath it.

## What dSense is not

- It is not metaphysics or a claim of magical awareness.
- It is not camera-based computer vision.
- It is not a microphone, RF, or thermal probe in v0.
- It is not initially a cryptographic random-number generator.
- It is not a full AI model yet.

## Core theory

In practice, a computer is physical, thermal, electrical, scheduled, and noisy. dSense treats that ordinary noise as a substrate field. Random-looking variation is not only an obstacle; with controlled labels, some events may appear as structured deformations in timing, latency, and process behavior.

## RAW and MIX lanes

Every v0 frame is 64 bytes. The RAW lane preserves inspectable timing structure such as clock delta, sleep drift, and a simple process activity estimate. The MIX lane is a 16-byte BLAKE2s-derived mixed representation of header and RAW bytes. v0 MIX is useful as a lower-bias representation for experiments, but it is not secure randomness and must not be used as cryptographic entropy.

## Scene-conditioned substrate learning

The first implementation is the Scene Wizard. Its job is to collect repeatable labeled scenes: idle baseline, left-to-right walk-by, phone near computer, door open/close, CPU load with no person, and similar controlled events. These datasets can later support anomaly detection, classifiers, contrastive learning, and scene-conditioned substrate models.

## Anomaly detection direction

A future always-on dSense Watcher could learn a normal machine/room baseline, detect anomalies, save rolling buffers around unusual periods, ask the user for labels, grow a local real-world dataset, and train small orbiters or classifiers. The user remains in control of labels and sharing.

## Latent orbiters

Latent orbiters are future small interpreter models that sit close to substrate channels. They would summarize local timing and environment events for larger AI systems, allowing the larger model to receive compact statements such as “scheduler jitter increased,” “room-like perturbation detected,” or “baseline degraded.”

Current orbiters emit structured local evidence summaries rather than vague claims. They include typed sections for timing, activity, drift, privacy, and transfer, local adapter status, and confidence disclaimers.

## Current v1 scope: local dSense console

v1 provides a terminal-based Python tool with:

- project initialization under `datasets/<project_name>/`
- channel scanning
- baseline recording
- guided scene recording with pre-roll, action, and post-roll timing
- a full-screen TUI phase dashboard for recording, learning, classification, channels, watcher scans, orbiter summaries, and transfer bundles
- fixed 64-byte binary frames
- automatic and manual event markers
- scene metadata
- preview CSV files
- baseline/anomaly model and deterministic scene classifier
- strict portable channel adapters for CPU load, disk latency, optional network latency, and power state
- local watcher, orbiter, and transfer artifacts

No microphone, camera, RF, broad environmental sensing, or external AI calls are implemented in v1. AI/orbiter enrichment is local-only and can be pointed at an embedded Gemma 4 Edge runtime.

## Install and run

Requires Python 3.11+.

```bash
python -m pip install -e .
python -m dsense doctor
python -m dsense init demo_lab
python -m dsense scan
python -m dsense scan --advanced
python -m dsense record-baseline demo_lab --duration 30
python -m dsense tui --label person_walks_front_left_to_right --duration 10 --pre-roll 2 --action 5 --post-roll 3 --repeat 3
python -m dsense scene demo_lab --label person_walks_front_left_to_right --duration 10 --pre-roll 2 --action 5 --post-roll 3 --repeat 3
python -m dsense scene demo_lab --label linux_activity --duration 5 --channels portable,linux --yes
python -m dsense train-baseline
python -m dsense train-classifier
python -m dsense extract-features base
python -m dsense rank-channels base
python -m dsense evaluate-scenes
python -m dsense export-transfer
python -m dsense list-scenes demo_lab
python -m dsense export-preview demo_lab
```

If installed with the console script, replace `python -m dsense` with `dsense`.

The default tick rate is 100 Hz. Higher rates are allowed with `--tick-hz`, but 1000 Hz can be unrealistic in Python depending on the OS, scheduler, and machine state.

For non-interactive captures, `dsense scene` also provides `--yes` to keep captures without prompting.

`dsense doctor` checks Python version, dataset folder state, terminal/TUI support, write permissions, and channel availability. Use `python -m dsense validate base --verbose` before training or pass `--require-valid` to training/export commands so broken captures fail clearly.

Channel groups keep dSense portable by default. `portable` is the default group. `linux` adds optional `/proc` and thermal/sysfs adapters when readable. `experimental` currently reports the eBPF adapter as unavailable unless a future implementation is installed. Use `dsense scan --advanced` to see group and permission status, and `dsense scene base --channels portable,linux ...` to opt into Linux telemetry for a capture.

## TUI interaction recorder

Use `python -m dsense` or `python -m dsense tui` for the full-screen recorder. By default it opens the base project at `datasets/base/`, loads all existing scenes from that project, and stores new captures there. To use another project, pass it explicitly, for example `python -m dsense tui demo_lab`.

The TUI starts with an editable capture setup, shows detected channels, baseline/classifier status, the seven phase dashboard panels, and existing project scenes. It trains or refreshes local baseline/classifier models from accepted scenes, then records with a live overview of frame progress, current phase, timing drift, process estimate, and marker count. The setup screen includes preset groups for `user`, `baseline`, and `activity` scenes.

The baseline model is stored at `datasets/<project_name>/exports/baseline_model.json`. The classifier is stored at `datasets/<project_name>/exports/classifier.json`. They use accepted scene previews to build baseline channel profiles and label summaries. They are retrained automatically when the TUI opens and after new accepted recordings are added, so automatic event detection improves as the project grows. You can retrain them explicitly with:

```bash
python -m dsense train-baseline --require-valid
python -m dsense train-classifier --require-valid
```

## Phase 7 evaluation

The main scientific checkpoint is repeatability: repeated takes of the same scene should look similar, and different labels should separate. Record repeated takes under stable labels, then run:

```bash
python -m dsense evaluate-scenes base
python -m dsense evaluate-scenes base --out datasets/base/exports/evaluation_report.json
python -m dsense inspect-scene base scene_000001
python -m dsense inspect-frame base scene_000001 --tick 50
python -m dsense classify-scene base scene_000001
python -m dsense replay-scene base scene_000001
python -m dsense export-scene-json base scene_000001 --out datasets/base/exports/scene_000001.json
python -m dsense export-trace base scene_000001
python -m dsense view-scene base scene_000001
```

`evaluate-scenes` writes `datasets/<project_name>/exports/evaluation_report.json` unless `--out` is provided. It prints label counts, within-label similarity, between-label distance, a leave-one-out confusion matrix, baseline drift, label-pair distances, channel usefulness, and research answers for whether idle separates from activity, interactions separate from each other, which channel is carrying signal, and which labels need review. Baseline and classifier training dynamically discover numeric preview columns, so extra channels such as `cpu_load_ppm`, `disk_stat_latency_ns`, `network_latency_ns`, `power_online`, and `battery_percent` can contribute when present.

Use `python -m dsense extract-features base` to write `datasets/base/exports/features.json`, including the feature manifest used by training. Per-channel features include median, MAD, p95, max, variance, and slope. Use `python -m dsense rank-channels base` to print the most useful channels and the best feature behind each channel score.

`view-scene` writes a self-contained HTML scene viewer with timeline tracks for timing/process/system channels and an event rail for scene/action/manual/heuristic markers. `export-trace` writes `trace.json` in a trace-style structure with scene metadata, channel tracks, points, and events for future external viewers.

Recommended repeatability set:

- `baseline_idle`
- `typing_burst`
- `mouse_activity`
- `phone_near_computer`
- `walk_by`
- optional controls: `cpu_load_no_person`, `door_open_close`

On the setup screen:

- `1`-`7` switches between phase panels: Record, Learn, Classify, Channels, Watcher, Orbiters, Transfer
- `m` cycles scene mode: user interactions, baseline system scenes, and system activity scenes
- `p` / `o` cycles presets inside the current mode
- `g` toggles batch recording for the whole current preset group
- `a` toggles automatic heuristic event detection
- `t` retrains baseline and classifier models from accepted project scenes
- `v` validates the project and summarizes health
- `w` runs a local watcher scan and writes watcher/orbiter artifacts
- `e` exports a local transfer bundle
- `Enter` edits the selected field or cycles mode/preset/toggle fields
- `c` starts recording
- `q` exits the TUI from the setup screen

During recording:

- `scene_start`, `action_start`, `action_end`, and `scene_end` are recorded automatically and shown in the live event list
- `heuristic_signal_spike` events are generated automatically when the built-in signal watcher sees a timing/process deviation
- `SPACE` writes a `user_interaction_marker`
- `n` writes a `noise_marker`
- `q` writes a `review_flag`

The live view shows the total event count, automatic/manual marker counts, a signal watcher meter, an event rail, and the newest recorded events. The event rail uses `S/A/E/X` for scheduled system events, `!` for heuristic signal events, and `*` for manual markers. Markers are saved to `events.jsonl` alongside `scene_start`, `action_start`, `action_end`, and `scene_end`. After each take, the TUI shows quality and frame counts and lets you keep, retake, or discard the recording. When a recording session completes, any key returns to the setup screen with the refreshed scene list. Existing scripted flows remain available through `record-baseline` and `scene`; `scene --tui` opens the same full-screen recorder using the scene command's arguments.

Watcher scans save candidate scenes and append `watcher/events.jsonl`. Rolling watcher mode keeps recent frames in memory and only saves a pre/post anomaly window when the detector triggers:

```bash
python -m dsense watcher base --rolling --pre 5 --post 10
python -m dsense watcher base --rolling --pre 5 --post 10 --duration 60
python -m dsense label-candidate base scene_000123 --label door_open_close
```

Rolling watcher sessions are appended to `watcher/sessions.jsonl`; saved anomaly windows use normal scene files, start as `watcher_anomaly_candidate`, and can be accepted by labeling them. Orbiter summaries are written to `exports/orbiters/summaries.jsonl`. Transfer bundles are written to `exports/transfer_bundle.json` and can be compared with:

```bash
python -m dsense compare-transfer datasets/base/exports/transfer_bundle.json
```

## Embedded Gemma 4 Edge

dSense can enrich local orbiter summaries with an embedded Gemma 4 Edge runner. This is optional and disabled unless configured. dSense does not download a model or call a remote API; it sends a compact prompt to a local command over stdin and reads the summary from stdout.

```bash
python -m dsense gemma-status
```

If the LiteRT-LM model has been imported as `dsense-gemma-4-edge`, dSense enables it automatically. To use a different local runner or model path, set:

```bash
export DSENSE_GEMMA_CMD="/path/to/your/local/gemma-edge-runner --model /path/to/gemma --prompt {prompt}"
export DSENSE_GEMMA_MODEL="gemma-4-edge"
python -m dsense gemma-status
```

If your local runner reads prompts from stdin, omit `{prompt}` and dSense will pipe the prompt to the command instead.

When enabled, watcher/orbiter summaries include `gemma_edge` metadata and `gemma_summary`. The TUI Orbiters phase shows whether Gemma Edge is on or off.

Structured orbiter commands:

```bash
python -m dsense orbiter-run base scene_000001
python -m dsense orbiter-evaluate base
```

`orbiter-run` summarizes one scene with stable JSON fields for downstream tools. `orbiter-evaluate` compares orbiter/classifier summary labels against actual user labels so evidence quality stays measurable. Local adapter metadata is explicit: Gemma is optional, the deterministic tiny classifier is local, ONNX is a placeholder, and remote calls are disabled.

## Dataset format

Each project is stored under:

```text
datasets/<project_name>/
  manifest.json
  channels.json
  scenes/
    scene_000001/
      scene.json
      frames.ds64
      events.jsonl
      preview.csv
      notes.txt
      checksum.txt
  exports/
    baseline_model.json
    classifier.json
    transfer_bundle.json
    orbiters/
      summaries.jsonl
  watcher/
    events.jsonl
```

`scene.json` stores label, time windows, channel metadata, quality summary, acceptance state, and notes. `frames.ds64` contains only 64-byte frames. `events.jsonl` stores scene start, action start, action end, scene end, manual markers, and heuristic events. `preview.csv` exposes inspectable columns: `tick`, `t_ns`, `dt_ns`, `sleep_drift_ns`, `process_ns_estimate`, `quality_flags`, plus optional extra channel columns such as `cpu_load_ppm`, `disk_stat_latency_ns`, `network_latency_ns`, `power_online`, and `battery_percent`. `checksum.txt` stores a SHA-256 checksum of the frame file.

## Frame format summary

| Bytes | Field |
| --- | --- |
| 0-3 | magic/version `DS01` |
| 4-7 | sequence number, uint32 little-endian |
| 8-15 | monotonic timestamp ns, uint64 little-endian |
| 16-19 | channel availability mask, uint32 |
| 20-23 | channel quality/degradation mask, uint32 |
| 24-39 | RAW lane: `dt_ns`, `sleep_drift_ns`, `process_ns_estimate`, reserved |
| 40-55 | MIX lane: BLAKE2s-derived 16-byte mix |
| 56-63 | checksum: truncated BLAKE2s digest over bytes 0-55 |

See `docs/frame-format.md` for the exact layout.

## Research roadmap status

- Phase 0: Scene Wizard and dataset format — implemented.
- Phase 1 stabilization: CI, basic Ruff linting, `doctor`, validation gates, clearer CLI errors, and sample dataset tests — implemented.
- Phase 1: baseline/anomaly learner — implemented as a local robust baseline model and TUI signal watcher.
- Phase 2: simple scene classifier — implemented as a deterministic nearest-profile classifier.
- Phase 3: dynamic feature extraction and richer channel adapters — implemented with numeric preview discovery, feature manifests, channel ranking, strict portable adapters, optional Linux adapters, adapter groups, and graceful degradation.
- Phase 4: local always-on dSense Watcher — implemented with TUI scans plus rolling anomaly windows, cooldowns, session logs, and candidate labeling.
- Phase 5: orbiters and AI integration — implemented as structured local evidence summaries with timing/activity/drift/privacy/transfer sections, local adapter metadata, confidence disclaimers, and summary-vs-label evaluation.
- Phase 6: cross-machine and cross-room transfer tests — implemented as local transfer bundle export/compare.
- Phase 7: repeatability and evaluation — implemented with dynamic preview features, evaluation reports, validation gates, and replay/debug commands.

## Privacy and safety

Substrate signals can become fingerprints of a machine, room, routine, or user behavior. dSense does not collect microphone or camera data by default. Labels and metadata may still reveal private behavior. Keep datasets local unless intentionally shared, and do not overinterpret v0 signals.

Use the privacy tooling before sharing:

```bash
python -m dsense privacy-report base
python -m dsense export-transfer base --redact
```

`privacy-report` flags sensitive labels, free-form notes, timestamps, repeated scene counts, and channels that can reveal power, thermal, network, or process state. `export-transfer --redact` writes a safe transfer bundle with model statistics and channel availability while removing project name, timestamps, label profiles, label counts, notes, and raw scenes. Sharing raw `datasets/<project>/scenes/` folders should be treated as intentional disclosure.

## Design principles

- universal first
- graceful degradation
- fixed frame format
- record/replay/null modes for reproducibility
- no overclaiming
- labeled data before model complexity
- boring tools before exotic probes
