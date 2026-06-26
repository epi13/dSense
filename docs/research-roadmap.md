# dSense research roadmap

## Controlled scene capture

Phase 0 focuses on repeatable labeled scenes. The Scene Wizard records pre-roll, action, and post-roll windows so later tools can compare baseline substrate behavior against known perturbations.

## Anomaly detection

Phase 1 should learn a local baseline from idle and normal-room captures. Anomalies should first be treated as prompts for labeling, not as proof of a specific real-world event.

## Always-on baseline learning

A future watcher can maintain rolling buffers, detect deviations, persist short windows, and ask the user whether an event was meaningful. Record/replay/null modes are required for reproducibility.

## Orbiters

Orbiters are small local interpreters that convert raw substrate streams into compact event summaries for larger AI systems. They should report confidence, degradation, and channel availability.

## Future sensor channels

Potential channels include thermal telemetry, battery and power state, network latency, disk IO latency, fan state, audio/RF-derived features, and hardware counters. These should degrade gracefully and remain optional.

## Strict substrate mode vs broad substrate mode

Strict substrate mode uses only intrinsic machine/OS behavior such as timing, scheduler, cache, IO, and power-state signals. Broad substrate mode may include optional environmental adapters such as audio, RF, or thermal sensors. v0 stays close to strict substrate mode and avoids heavy platform-specific probes.
