from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GEMMA_MODEL_ID = "dsense-gemma-4-edge"


@dataclass(frozen=True)
class GemmaEdgeConfig:
    command: str
    model: str = "gemma-4-edge"
    timeout_s: float = 8.0

    @property
    def enabled(self) -> bool:
        return bool(self.command.strip())


def load_gemma_edge_config() -> GemmaEdgeConfig:
    command = os.environ.get("DSENSE_GEMMA_CMD", "")
    model = os.environ.get("DSENSE_GEMMA_MODEL", DEFAULT_GEMMA_MODEL_ID)
    if not command and os.environ.get("DSENSE_GEMMA_DISABLE", "").lower() not in {"1", "true", "yes"}:
        command = default_litert_lm_command(model)
    return GemmaEdgeConfig(
        command=command,
        model=model,
        timeout_s=float(os.environ.get("DSENSE_GEMMA_TIMEOUT", "8")),
    )


def default_litert_lm_command(model: str = DEFAULT_GEMMA_MODEL_ID) -> str:
    executable = shutil.which("litert-lm")
    local_executable = Path.home() / ".local" / "bin" / "litert-lm"
    if executable is None and local_executable.exists():
        executable = str(local_executable)
    model_file = Path.home() / ".litert-lm" / "models" / model / "model.litertlm"
    if executable and model_file.exists():
        return f"{shlex.quote(executable)} run {shlex.quote(model)} --prompt {{prompt}}"
    repo_model = Path("models/gemma-4-edge/gemma-4-E2B-it-web.litertlm")
    if executable and repo_model.exists():
        return f"{shlex.quote(executable)} run {shlex.quote(str(repo_model))} --prompt {{prompt}}"
    return ""


def gemma_edge_status(config: GemmaEdgeConfig | None = None) -> dict[str, object]:
    config = config or load_gemma_edge_config()
    return {
        "enabled": config.enabled,
        "model": config.model,
        "command": config.command,
        "mode": "local_command_prompt_template" if "{prompt}" in config.command else "local_command_stdin",
        "timeout_s": config.timeout_s,
    }


def build_orbiter_prompt(summary: dict[str, object]) -> str:
    payload = {
        "scene_id": summary.get("scene_id"),
        "baseline_status": summary.get("baseline_status", {}),
        "classifier_prediction": summary.get("classifier_prediction", {}),
        "anomaly_score": summary.get("anomaly_score", 0.0),
        "channel_availability_mask": summary.get("channel_availability_mask", 0),
        "quality_flags": summary.get("quality_flags", 0),
    }
    return (
        "You are dSense's local Gemma 4 Edge orbiter. "
        "Summarize this machine-substrate event in one concise sentence. "
        "Do not claim camera, microphone, RF, or human certainty. "
        "Mention confidence and strongest signal only when present.\n"
        f"{json.dumps(payload, sort_keys=True)}"
    )


def enrich_orbiter_summary(summary: dict[str, object], config: GemmaEdgeConfig | None = None) -> dict[str, object]:
    config = config or load_gemma_edge_config()
    enriched = dict(summary)
    enriched["gemma_edge"] = gemma_edge_status(config)
    if not config.enabled:
        enriched["gemma_edge"]["used"] = False
        enriched["gemma_edge"]["reason"] = "DSENSE_GEMMA_CMD is not set"
        return enriched

    prompt = build_orbiter_prompt(summary)
    try:
        command = config.command.replace("{prompt}", shlex.quote(prompt))
        completed = subprocess.run(
            shlex.split(command),
            input=None if "{prompt}" in config.command else prompt,
            text=True,
            capture_output=True,
            timeout=config.timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        enriched["gemma_edge"]["used"] = False
        enriched["gemma_edge"]["reason"] = str(exc)
        return enriched

    output = clean_litert_lm_output(completed.stdout)
    enriched["gemma_edge"]["used"] = completed.returncode == 0 and bool(output)
    enriched["gemma_edge"]["returncode"] = completed.returncode
    if completed.stderr.strip():
        enriched["gemma_edge"]["stderr"] = completed.stderr.strip()[:500]
    if output:
        enriched["gemma_summary"] = output[:1000]
        enriched["summary"] = output[:1000]
    return enriched


def clean_litert_lm_output(output: str) -> str:
    noisy_prefixes = (
        "Using GPU backend",
        "Using CPU backend",
        "Using NPU backend",
        "Warning:",
    )
    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip() and not line.strip().startswith(noisy_prefixes)
    ]
    return "\n".join(lines).strip()
