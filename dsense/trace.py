from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from .replay import inspect_scene, read_events, read_preview_rows
from .utils.files import ensure_dir, write_json

DEFAULT_TRACKS = [
    "dt_ns",
    "sleep_drift_ns",
    "process_ns_estimate",
    "cpu_load_ppm",
    "disk_stat_latency_ns",
    "network_latency_ns",
    "power_online",
    "battery_percent",
]


def trace_path(scene_dir: Path) -> Path:
    return scene_dir / "trace.json"


def viewer_path(scene_dir: Path) -> Path:
    return scene_dir / "viewer.html"


def export_trace(project_name: str, scene_dir: Path, out_path: Path | None = None) -> dict[str, object]:
    preview_rows = read_preview_rows(scene_dir / "preview.csv")
    events = read_events(scene_dir)
    summary = inspect_scene(scene_dir)
    tracks = []
    for channel in _track_columns(preview_rows):
        points = []
        for row in preview_rows:
            value = _float_value(row.get(channel))
            if value is None:
                continue
            points.append({
                "tick": _int_value(row.get("tick"), len(points)),
                "t_ms": _row_time_ms(row, summary),
                "value": value,
            })
        if points:
            tracks.append({"name": channel, "unit": _unit_for(channel), "points": points})
    trace = {
        "format": "dsense-trace-v1",
        "project_name": project_name,
        "scene": summary,
        "duration_ms": summary.get("duration_ms", 0),
        "tracks": tracks,
        "events": [_trace_event(event) for event in events],
    }
    out = out_path or trace_path(scene_dir)
    ensure_dir(out.parent)
    write_json(out, trace)
    return trace


def write_scene_viewer(project_name: str, scene_dir: Path, out_path: Path | None = None, open_browser: bool = False) -> Path:
    trace = export_trace(project_name, scene_dir)
    out = out_path or viewer_path(scene_dir)
    ensure_dir(out.parent)
    data = json.dumps(trace, sort_keys=True)
    out.write_text(_html_document(data), encoding="utf-8")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def _track_columns(rows: list[dict[str, str]]) -> list[str]:
    present = {key for row in rows for key in row if _float_value(row.get(key)) is not None}
    ordered = [column for column in DEFAULT_TRACKS if column in present]
    ordered.extend(sorted(present - set(ordered) - {"tick", "t_ns", "quality_flags"}))
    return ordered


def _trace_event(event: dict[str, object]) -> dict[str, object]:
    return {
        "name": str(event.get("event", "event")),
        "t_ms": _int_value(event.get("t_ms"), 0),
        "source": str(event.get("source", "")),
        "data": event,
    }


def _row_time_ms(row: dict[str, str], summary: dict[str, object]) -> float:
    tick = _int_value(row.get("tick"), 0)
    tick_hz = float(summary.get("tick_hz", 0) or 0)
    if tick_hz > 0:
        return round(tick * 1000.0 / tick_hz, 3)
    return float(tick)


def _unit_for(channel: str) -> str:
    if channel.endswith("_ns") or channel in {"dt_ns", "sleep_drift_ns", "process_ns_estimate"}:
        return "ns"
    if channel.endswith("_ppm"):
        return "ppm"
    if channel.endswith("_percent"):
        return "%"
    return ""


def _float_value(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: object, default: int) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _html_document(trace_json: str) -> str:
    safe_json = trace_json.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dSense Scene Viewer</title>
<style>
body {{ margin: 0; font: 14px system-ui, sans-serif; color: #202124; background: #f7f8fa; }}
header {{ padding: 16px 20px; background: #ffffff; border-bottom: 1px solid #d9dde3; position: sticky; top: 0; z-index: 2; }}
h1 {{ margin: 0 0 6px; font-size: 20px; }}
main {{ padding: 16px 20px 28px; }}
.meta {{ color: #5f6670; display: flex; gap: 16px; flex-wrap: wrap; }}
.track {{ background: #fff; border: 1px solid #d9dde3; margin: 12px 0; padding: 10px; border-radius: 6px; }}
.track h2 {{ margin: 0 0 8px; font-size: 14px; }}
svg {{ width: 100%; height: 96px; display: block; background: #fbfcfd; border: 1px solid #edf0f3; }}
.events {{ background: #fff; border: 1px solid #d9dde3; padding: 10px; border-radius: 6px; }}
.event {{ display: inline-block; margin: 4px 6px 4px 0; padding: 3px 6px; border-radius: 4px; background: #eef3ff; border: 1px solid #c9d8ff; }}
</style>
</head>
<body>
<header>
  <h1 id="title">dSense Scene Viewer</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <section class="events"><strong>Event rail</strong><div id="events"></div></section>
  <section id="tracks"></section>
</main>
<script id="trace-data" type="application/json">{safe_json}</script>
<script>
const trace = JSON.parse(document.getElementById('trace-data').textContent);
const scene = trace.scene || {{}};
document.getElementById('title').textContent = `${{scene.scene_id || 'scene'}} · ${{scene.label || 'unknown'}}`;
document.getElementById('meta').innerHTML = [
  `duration ${{trace.duration_ms || 0}}ms`,
  `frames ${{scene.frame_count || 0}}`,
  `events ${{(trace.events || []).length}}`,
  `accepted ${{scene.accepted}}`
].map(x => `<span>${{x}}</span>`).join('');
const events = document.getElementById('events');
(trace.events || []).forEach(ev => {{
  const el = document.createElement('span');
  el.className = 'event';
  el.textContent = `${{ev.t_ms}}ms ${{ev.name}}`;
  events.appendChild(el);
}});
const tracks = document.getElementById('tracks');
for (const track of trace.tracks || []) {{
  const wrap = document.createElement('div');
  wrap.className = 'track';
  wrap.innerHTML = `<h2>${{track.name}} ${{track.unit ? '(' + track.unit + ')' : ''}}</h2>`;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 1000 100');
  const points = track.points || [];
  const maxT = Math.max(1, ...(points.map(p => p.t_ms)));
  const vals = points.map(p => p.value);
  const minV = Math.min(...vals, 0);
  const maxV = Math.max(...vals, 1);
  const span = Math.max(1, maxV - minV);
  const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  poly.setAttribute('fill', 'none');
  poly.setAttribute('stroke', '#1769aa');
  poly.setAttribute('stroke-width', '2');
  poly.setAttribute('points', points.map(p => `${{(p.t_ms / maxT) * 1000}},${{92 - ((p.value - minV) / span) * 84}}`).join(' '));
  svg.appendChild(poly);
  for (const ev of trace.events || []) {{
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    const x = (ev.t_ms / maxT) * 1000;
    line.setAttribute('x1', x); line.setAttribute('x2', x); line.setAttribute('y1', 0); line.setAttribute('y2', 100);
    line.setAttribute('stroke', '#d94f45'); line.setAttribute('stroke-width', '1');
    svg.appendChild(line);
  }}
  wrap.appendChild(svg);
  tracks.appendChild(wrap);
}}
</script>
</body>
</html>
"""
