from __future__ import annotations

import pytest

from dsense.autotest import validate_dataset
from dsense.cli import main
from dsense.doctor import doctor_ok, run_doctor
from dsense.manifest import DEFAULT_PROJECT
from dsense.transfer import transfer_bundle_path
from dsense.utils.files import read_json


def test_sample_dataset_fixture_validates(sample_dataset):
    result = validate_dataset("base")

    assert sample_dataset.name == "base"
    assert result.error_count == 0
    assert result.valid_scenes == 2
    assert result.comparison["labels"]["baseline_idle"] == 1


def test_doctor_reports_fresh_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    main(["doctor"])

    out = capsys.readouterr().out
    assert "dSense doctor" in out
    assert "Python" in out
    assert "dataset" in out
    assert doctor_ok(run_doctor("base"))


def test_validate_missing_project_has_clear_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        main(["validate", "missing_project", "--verbose"])

    assert "Not found:" in str(exc.value)
    assert "No scenes directory" in str(exc.value)


def test_require_valid_bad_dataset_message(sample_dataset, capsys):
    bad_preview = sample_dataset / "scenes" / "scene_000001" / "preview.csv"
    bad_preview.write_text("tick,t_ns\n0,1\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["train-classifier", "base", "--require-valid"])

    out = capsys.readouterr().out
    assert "Validation failed for base" in str(exc.value)
    assert "Missing CSV columns" in out


def test_privacy_report_and_redacted_export_commands(sample_dataset, capsys):
    main(["privacy-report", DEFAULT_PROJECT])
    report_out = capsys.readouterr().out
    assert "Privacy report: base" in report_out
    assert "Warnings:" in report_out

    main(["export-transfer", DEFAULT_PROJECT, "--redact"])
    export_out = capsys.readouterr().out
    bundle = read_json(transfer_bundle_path(DEFAULT_PROJECT))

    assert "Sharing summary before export:" in export_out
    assert bundle["redacted"] is True
    assert bundle["sharing_summary"]["contains_raw_scenes"] is False


def test_orbiter_commands(sample_dataset, monkeypatch, capsys):
    monkeypatch.setenv("DSENSE_GEMMA_DISABLE", "1")

    main(["orbiter-run", DEFAULT_PROJECT, "scene_000001"])
    run_out = capsys.readouterr().out
    assert '"schema_version": "dsense-orbiter-v1"' in run_out
    assert '"summary_comparison"' in run_out

    main(["orbiter-evaluate", DEFAULT_PROJECT, "--limit", "1"])
    eval_out = capsys.readouterr().out
    assert "Orbiter evaluation: base" in eval_out
    assert "Evaluated:" in eval_out
