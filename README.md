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
python -m dsense baseline-suite base --target-scenes 200 --yes
python -m dsense auto-scenes base --group baseline --repeat 3 --yes
python -m dsense auto-scenes base --group activity --repeat 2 --yes
python -m dsense tui --label person_walks_front_left_to_right --duration 10 --pre-roll 2 --action 5 --post-roll 3 --repeat 3
python -m dsense scene demo_lab --label person_walks_front_left_to_right --duration 10 --pre-roll 2 --action 5 --post-roll 3 --repeat 3
python -m dsense scene demo_lab --label linux_activity --duration 5 --channels portable,linux --yes
python -m dsense train-baseline
python -m dsense train-classifier
python -m dsense train-timeseries
python -m dsense update-intelligence base
python -m dsense council-status base
python -m dsense extract-features base
python -m dsense rank-channels base
python -m dsense evaluate-scenes
python -m dsense export-transfer
python -m dsense list-scenes demo_lab
python -m dsense export-preview demo_lab
```

If installed with the console script, replace `python -m dsense` with `dsense`.

The default tick rate is 100 Hz. Higher rates are allowed with `--tick-hz`, but 1000 Hz can be unrealistic in Python depending on the OS, scheduler, and machine state. The recorder respects each channel's declared `rate_hz`: lower-rate channels are sampled only when due and their last value is reused between samples, with sampled/stale/unavailable masks written to `preview.csv`.

For non-interactive captures, `dsense scene` also provides `--yes` to keep captures without prompting.

## Scene presets and automatic capture

dSense presets are split into three groups:

- `user`: manual interaction labels such as approach direction, walk-by direction, sitting down, leaving, typing, mouse use, phone placement, door/object movement, and table taps. These require a person to perform the action during the action window.
- `baseline`: automatic recordings of "nothing intentional is happening" states such as quiet idle, screen-on idle, charging, battery, network-on, post-startup, and warmed-up idle.
- `activity`: controlled machine-internal labels such as bounded CPU work, temp-file metadata or write activity, modest memory allocation, deterministic Python loops, mixed CPU/disk work, and a no-op automation control.

Manual capture remains available through the TUI or guided scene command:

```bash
python -m dsense tui base
python -m dsense scene base --label user_walk_left_to_right --duration 10 --pre-roll 2 --action 5 --post-roll 3
```

Baseline and activity presets can be captured without user interaction:

```bash
python -m dsense baseline-suite base --target-scenes 200 --yes
python -m dsense baseline-suite base --target-scenes 100 --repeat 2 --yes
python -m dsense baseline-suite base --target-scenes 200 --seed 42 --yes
python -m dsense baseline-suite base --target-scenes 200 --categories idle,cpu,disk,proc --yes
python -m dsense baseline-suite base --target-scenes 200 --dry-run
python -m dsense auto-scenes base --group baseline --yes
python -m dsense auto-scenes base --group activity --yes
python -m dsense auto-scenes base --group baseline --repeat 3 --tick-hz 100 --yes
python -m dsense auto-scenes base --include baseline_idle_quiet,activity_cpu_light --yes
python -m dsense auto-scenes base --exclude activity_disk_write_tempfile --yes
```

Activity workloads run only during the configured action window. They are pure Python standard-library helpers, use local temporary files when disk activity is needed, clean up afterward, and do not use microphone, camera, RF, network, or external AI calls. Treat these scenes as controlled machine-internal labels, not proof of external sensing.

### Baseline suite

`baseline-suite` is the recommended way to build a deeper negative-control dataset on Linux. It generates a deterministic plan from a structured catalog of safe automatic controls: quiet idle, timing/scheduler observation, modest CPU loops, memory allocation/release, temp-file metadata/read/write controls, Linux `/proc` and sysfs reads when available, mixed tiny workloads, and longitudinal drift repeats. These scenes are labeled as `baseline_...` controls and marked as no intentional physical interaction.

By default, network controls are disabled. They are included only with `--include-network` and only when `DSENSE_NET_HOST` is explicitly configured. Heavier workloads are disabled unless `--include-heavy` is passed. The suite writes normal scene artifacts plus `datasets/<project>/exports/baseline_suite_report.json`, including category counts, scenario order, seed, channel groups, failures/skips, validation summary, baseline/classifier summaries, drift, and noisy-channel summary.

Dry-run planning shows the suite without recording:

```bash
python -m dsense baseline-suite base --target-scenes 200 --dry-run
```

More baseline/control scenes reduce false positives from ordinary machine variation, but they do not prove physical sensing or reliable human detection.

Recommended first dataset sequence:

```bash
python -m dsense init base
python -m dsense baseline-suite base --target-scenes 200 --yes
python -m dsense train-baseline base
python -m dsense train-classifier base
python -m dsense train-timeseries base
python -m dsense update-intelligence base
python -m dsense validate base --verbose
python -m dsense evaluate-scenes base
python -m dsense tui base
```

`dsense doctor` checks Python version, dataset folder state, terminal/TUI support, write permissions, and channel availability. Use `python -m dsense validate base --verbose` before training or pass `--require-valid` to training/export commands so broken captures fail clearly.

Channel groups keep dSense portable by default. `portable` is the default group. `linux` adds optional `/proc` and thermal/sysfs adapters when readable. `experimental` currently reports the eBPF adapter as unavailable unless a future implementation is installed. Use `dsense scan --advanced` to see group and permission status, and `dsense scene base --channels portable,linux ...` to opt into Linux telemetry for a capture. The optional network latency channel is disabled unless `DSENSE_NET_HOST` is set; it performs TCP connect timing and can perturb measurements, so use it deliberately.

## TUI interaction recorder

Use `python -m dsense` or `python -m dsense tui` for the full-screen recorder. By default it opens the base project at `datasets/base/`, loads all existing scenes from that project, and stores new captures there. To use another project, pass it explicitly, for example `python -m dsense tui demo_lab`.

The TUI is Linux/Unix-first because it depends on terminal curses behavior. Non-TUI commands such as `doctor`, `scan`, `init`, `record-baseline`, `scene`, `validate`, and export commands are intended to remain portable where the underlying channel adapters are available.

When the TUI opens normally, dSense shows a startup/update pipeline and refreshes the local intelligence stack before opening the main dashboard. The update coordinates dataset validation, baseline training, deterministic classifier training, the time-series model, evaluation, watcher state, orbiter summaries, transfer status, and the shared Council artifact at `datasets/<project>/exports/intelligence_state.json`. Failures in one layer are shown as warnings or failed steps so the project can still open when possible.

To open the TUI without startup training, watcher scans, orbiters, or baseline-suite work:

```bash
dsense tui base --no-startup-intelligence
dsense tui-safe base
```

Granular startup controls:

```bash
dsense tui base --no-auto-baseline
dsense tui base --auto-baseline-policy off
dsense tui base --auto-baseline-policy missing-only
dsense tui base --auto-baseline-policy startup
dsense tui base --force-auto-baseline
dsense tui base --auto-baseline-duration 10
dsense tui base --no-startup-suite
dsense tui base --no-startup-watchers
dsense tui base --no-startup-orbiters
dsense tui base --no-startup-training
dsense tui base --startup-suite-target 200
dsense tui base --startup-suite-duration 0.2
```

The TUI opens as a tabbed local control panel. The tab bar includes `Record`, `Scenes`, `Channels`, `Council`, `Learn`, `Classify`, `Evaluation`, `Watcher`, `Orbiters`, `Transfer`, `Validate`, and `Help`. The `Record` tab keeps the editable capture setup with preset groups for `user`, `baseline`, and `activity` scenes. `Scenes` lets you browse recorded scenes and inspect wrapped scene notes. `Channels` shows adapter status. `Council` shows coordinated local evidence, model agreement, confidence, warnings, and recommendations. `Learn` and `Classify` show local model state. `Watcher`, `Orbiters`, `Transfer`, and `Validate` expose project artifacts without leaving the terminal.

Press `u` from the TUI setup screen to run **Update Intelligence** again. This is the primary manual update path and refreshes validation, baseline, classifier, time-series, evaluation, watcher/orbiter context, transfer, and Council state. Focused commands such as validation and export remain available, but the main workflow is one comprehensive local update rather than separate model operations.

The baseline model is stored at `datasets/<project_name>/exports/baseline_model.json`. The classifier is stored at `datasets/<project_name>/exports/classifier.json`. The time-series model is stored at `datasets/<project_name>/exports/timeseries_model.json`. These use accepted scene previews to build baseline channel profiles, label summaries, and temporal profiles. You can refresh them explicitly with:

```bash
python -m dsense train-baseline --require-valid
python -m dsense train-classifier --require-valid
python -m dsense train-timeseries --require-valid
python -m dsense update-intelligence base
python -m dsense council-status base
```

### Intelligence Council and time-series model

The Intelligence Council is a coordination/reporting layer, not a claim of awareness. It combines coordinated local evidence layers: baseline readiness, deterministic classifier labels, time-series profile agreement, watcher events, orbiter summary availability, evaluation quality, baseline drift, useful channels, and transfer status. It writes inspectable JSON to `exports/intelligence_state.json` with confidence, warnings, recommendations, best channels, weak labels, and per-step status.

The time-series model is dependency-light and standard-library based. It reads accepted `preview.csv` files, discovers numeric channels dynamically, extracts temporal features such as first/last value, slope, peak count, roughness, rolling variance, max absolute delta, and window medians, then predicts by normalized distance to per-label profiles. It is additive and does not replace the deterministic classifier.

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

- `TAB` switches to the next tab
- `Shift+TAB` or left arrow switches to the previous tab
- `1`-`0` jumps directly to tabs from `Record` through `Help`
- `m` cycles scene mode: user interactions, baseline system scenes, and system activity scenes
- `p` / `o` cycles presets inside the current mode
- `g` toggles batch recording for the whole current preset group
- `a` toggles automatic heuristic event detection
- `u` updates the full local intelligence stack
- `v` validates the project and summarizes health
- `e` exports a local transfer bundle
- `Enter` edits the selected field or cycles mode/preset/toggle fields
- `c` starts recording from the `Record` tab
- `q` exits the TUI from the setup screen

The `Help` tab summarizes the recommended workflow, scene-mode meanings, and the pre-roll/action/post-roll timing model. dSense remains a local substrate-signal capture tool; labels and summaries are evidence for controlled experiments, not claims of external certainty.

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

`scene.json` stores label, time windows, channel metadata, quality summary, acceptance state, and notes. `frames.ds64` contains only 64-byte frames. RAW int32 frame fields are clamped on overflow and marked through the quality mask rather than crashing a recording. `events.jsonl` stores scene start, action start, action end, scene end, manual markers, and heuristic events. `preview.csv` exposes inspectable columns: `tick`, `t_ns`, `dt_ns`, `sleep_drift_ns`, `process_ns_estimate`, `quality_flags`, channel scheduling masks, plus optional extra channel columns such as `cpu_load_ppm`, `disk_stat_latency_ns`, `network_latency_ns`, `power_online`, and `battery_percent`. `checksum.txt` stores a SHA-256 checksum of the frame file.

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
