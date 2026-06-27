# dSense dataset format

Projects live under `datasets/<project_name>/`.

```text
datasets/demo_lab/
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
    preview_index.csv
    baseline_model.json
    classifier.json
    transfer_bundle.json
    orbiters/
      summaries.jsonl
  watcher/
    events.jsonl
```

Example `scene.json`:

```json
{
  "scene_id": "scene_000001",
  "label": "person_walks_front_left_to_right",
  "created_utc": "2026-06-26T12:00:00Z",
  "duration_ms": 10000,
  "tick_hz": 100,
  "frame_size_bytes": 64,
  "mode": "record",
  "machine_state": {},
  "pre_roll_ms": 2000,
  "action_start_ms": 2000,
  "action_end_ms": 7000,
  "post_roll_ms": 3000,
  "channels": [{"id": "clock_delta", "available": true, "bit": 0}],
  "quality": {"confidence": 0.98},
  "accepted": true,
  "notes": "controlled walk-by"
}
```

Example `events.jsonl`:

```jsonl
{"t_ms":0,"event":"scene_start"}
{"t_ms":2000,"event":"action_start"}
{"t_ms":7000,"event":"action_end"}
{"t_ms":10000,"event":"scene_end"}
```

`preview.csv` is intentionally simple for inspection and plotting. It includes `tick`, `t_ns`, `dt_ns`, `sleep_drift_ns`, `process_ns_estimate`, and `quality_flags`. v1 may add optional strict-portable channel columns such as `cpu_load_ppm`, `disk_stat_latency_ns`, `network_latency_ns`, `power_online`, and `battery_percent`.

`baseline_model.json`, `classifier.json`, `transfer_bundle.json`, watcher events, and orbiter summaries are derived artifacts. They can be regenerated from accepted scenes and do not change the fixed 64-byte frame format.
