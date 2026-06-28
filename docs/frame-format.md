# dSense v0 frame format

Each `frames.ds64` file is a concatenation of fixed 64-byte frames. If a file size is not divisible by 64, the file is invalid.

| Byte range | Size | Type | Field |
| --- | ---: | --- | --- |
| 0-3 | 4 | bytes | magic/version, `DS01` |
| 4-7 | 4 | uint32 LE | sequence number |
| 8-15 | 8 | uint64 LE | `time.perf_counter_ns()` timestamp |
| 16-19 | 4 | uint32 LE | channel availability mask |
| 20-23 | 4 | uint32 LE | channel quality/degradation mask |
| 24-27 | 4 | int32 LE | RAW `dt_ns` |
| 28-31 | 4 | int32 LE | RAW `sleep_drift_ns` |
| 32-35 | 4 | int32 LE | RAW `process_ns_estimate` |
| 36-39 | 4 | uint32 LE | RAW reserved |
| 40-55 | 16 | bytes | MIX lane, BLAKE2s-derived from header and RAW |
| 56-63 | 8 | bytes | truncated BLAKE2s checksum over bytes 0-55 |

Channel bit assignments in v0:

- bit 0: `clock_delta`
- bit 1: `sleep_jitter`
- bit 2: `process_probe`
- bit 3: `cpu_load`
- bit 4: `disk_latency`
- bit 5: `network_latency` (optional, disabled unless configured)
- bit 6: `power_state` (available when the OS exposes battery/power files)

The MIX lane is not cryptographic entropy. It is a deterministic digest-based representation intended to reduce obvious bias for downstream experiments while preserving honest documentation of its limits.

Only the original RAW fields are packed into the fixed 64-byte frame. Additional v1 channel values are stored in `preview.csv` and scene metadata so older frame readers remain compatible.

RAW int32 fields are saturated to the signed int32 range before packing. This prevents suspend/resume pauses, debugger stops, scheduler hiccups, or unusual tick settings from crashing frame writing. When the recorder sees an out-of-range RAW value it sets the high quality/degradation bit in the frame quality mask; older readers can still parse the clamped values.

Recorder channel scheduling respects each channel's declared `rate_hz`. The main recorder tick still runs at the scene `tick_hz`, but lower-rate channels are sampled only when due. Preview CSV rows include `channel_sampled_mask`, `channel_stale_mask`, and `channel_unavailable_mask` so reused stale values and unavailable channels remain inspectable without changing the fixed frame format.
