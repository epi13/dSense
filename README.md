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

## Current v0 scope: Python Scene Wizard

v0 provides a terminal-based Python tool with:

- project initialization under `datasets/<project_name>/`
- channel scanning
- baseline recording
- guided scene recording with pre-roll, action, and post-roll timing
- fixed 64-byte binary frames
- event markers
- scene metadata
- preview CSV files
- simple quality checks

No machine learning, microphone, camera, RF, thermal, or platform-specific hardware probes are implemented in v0.

## Install and run

Requires Python 3.11+.

```bash
python -m pip install -e .
python -m dsense init demo_lab
python -m dsense scan
python -m dsense record-baseline demo_lab --duration 30
python -m dsense scene demo_lab --label person_walks_front_left_to_right --duration 10 --pre-roll 2 --action 5 --post-roll 3 --repeat 3
python -m dsense list-scenes demo_lab
python -m dsense export-preview demo_lab
```

If installed with the console script, replace `python -m dsense` with `dsense`.

The default tick rate is 100 Hz. Higher rates are allowed with `--tick-hz`, but 1000 Hz can be unrealistic in Python depending on the OS, scheduler, and machine state.

For non-interactive captures, `dsense scene` also provides `--yes` to keep captures without prompting.

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
```

`scene.json` stores label, time windows, channel metadata, quality summary, acceptance state, and notes. `frames.ds64` contains only 64-byte frames. `events.jsonl` stores scene start, action start, action end, and scene end markers. `preview.csv` exposes inspectable columns: `tick`, `t_ns`, `dt_ns`, `sleep_drift_ns`, `process_ns_estimate`, and `quality_flags`. `checksum.txt` stores a SHA-256 checksum of the frame file.

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

## Research roadmap

- Phase 0: Scene Wizard and dataset format.
- Phase 1: baseline/anomaly learner.
- Phase 2: simple scene classifier.
- Phase 3: richer channel adapters.
- Phase 4: local always-on dSense Watcher.
- Phase 5: orbiters and AI integration.
- Phase 6: cross-machine and cross-room transfer tests.

## Privacy and safety

Substrate signals can become fingerprints of a machine, room, routine, or user behavior. dSense does not collect microphone or camera data by default. Labels and metadata may still reveal private behavior. Keep datasets local unless intentionally shared, and do not overinterpret v0 signals.

## Design principles

- universal first
- graceful degradation
- fixed frame format
- record/replay/null modes eventually
- no overclaiming
- labeled data before model complexity
- boring tools before exotic probes
