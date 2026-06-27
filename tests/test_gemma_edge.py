import sys

from dsense.gemma_edge import GemmaEdgeConfig, build_orbiter_prompt, clean_litert_lm_output, default_litert_lm_command, enrich_orbiter_summary, gemma_edge_status


def test_gemma_edge_status_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DSENSE_GEMMA_CMD", raising=False)
    monkeypatch.setenv("DSENSE_GEMMA_DISABLE", "1")

    status = gemma_edge_status()

    assert status["enabled"] is False


def test_default_litert_lm_command_detects_imported_model(tmp_path, monkeypatch):
    home = tmp_path / "home"
    exe = home / ".local" / "bin" / "litert-lm"
    model = home / ".litert-lm" / "models" / "dsense-gemma-4-edge" / "model.litertlm"
    exe.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "")

    command = default_litert_lm_command()

    assert "dsense-gemma-4-edge" in command
    assert "--prompt {prompt}" in command


def test_build_orbiter_prompt_is_privacy_bounded():
    prompt = build_orbiter_prompt({
        "scene_id": "scene_000001",
        "baseline_status": {"status": "anomaly", "channel": "dt_ns"},
        "classifier_prediction": {"label": "typing_burst", "confidence": 0.8},
        "anomaly_score": 7.2,
    })

    assert "Do not claim camera, microphone, RF, or human certainty" in prompt
    assert "typing_burst" in prompt


def test_enrich_orbiter_summary_uses_local_command():
    config = GemmaEdgeConfig(
        command=f"{sys.executable} -c \"import sys; sys.stdin.read(); print('local gemma summary')\"",
        timeout_s=2,
    )

    summary = enrich_orbiter_summary({"summary": "original", "scene_id": "scene_000001"}, config)

    assert summary["gemma_edge"]["used"] is True
    assert summary["summary"] == "local gemma summary"
    assert summary["gemma_summary"] == "local gemma summary"


def test_enrich_orbiter_summary_supports_prompt_template():
    config = GemmaEdgeConfig(
        command=f"{sys.executable} -c \"import sys; print('template gemma summary')\" {{prompt}}",
        timeout_s=2,
    )

    summary = enrich_orbiter_summary({"summary": "original", "scene_id": "scene_000001"}, config)

    assert summary["gemma_edge"]["used"] is True
    assert summary["gemma_edge"]["mode"] == "local_command_prompt_template"
    assert summary["summary"] == "template gemma summary"


def test_clean_litert_lm_output_removes_backend_noise():
    cleaned = clean_litert_lm_output("Using GPU backend for this model because CPU is unsupported.\nWarning: hello\nActual summary\n")

    assert cleaned == "Actual summary"
