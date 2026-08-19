"""The v7 archive must be complete, honest about its metric, and free of data."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from ecup_matching.build_submission_v7 import (
    build_model_metadata,
    build_submission_v7,
)


FOLD0_DIAGNOSTIC = 0.7023556010133556


def _model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "model_v7_teacher"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"\x00" * 64)
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    return model


def test_archive_contains_the_entrypoint_and_the_whole_closure(tmp_path):
    report = build_submission_v7(
        model_dir=_model_dir(tmp_path),
        output_zip=tmp_path / "out.zip",
        diagnostic_fold0=FOLD0_DIAGNOSTIC,
    )
    with zipfile.ZipFile(report["zip"]) as archive:
        names = set(archive.namelist())
    assert "metadata.json" in names
    assert "run.py" in names
    assert "model_v7_metadata.json" in names
    assert "model_v7_teacher/model.safetensors" in names
    for module in report["packaged_modules"]:
        assert module in names, f"{module} missing from the archive"


def test_archive_declares_the_organizer_contract(tmp_path):
    report = build_submission_v7(
        model_dir=_model_dir(tmp_path), output_zip=tmp_path / "out.zip"
    )
    with zipfile.ZipFile(report["zip"]) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata == {
        "image": "odsai/ecup26-matching-baseline:1.0",
        "entry_point": "python -u run.py",
    }


def test_fold0_diagnostic_is_never_promoted_to_a_strict_oof_score():
    metadata = build_model_metadata(
        diagnostic_fold0=FOLD0_DIAGNOSTIC,
        strict_oof=None,
        production_metrics=None,
    )
    assert metadata["diagnostic_fold0_macro_average_precision"] == FOLD0_DIAGNOSTIC
    assert metadata["strict_oof_macro_average_precision"] is None
    assert metadata["diagnostic_fold0_is_not_out_of_fold"] is True

    with pytest.raises(ValueError, match="must not repeat the fold-0 diagnostic"):
        build_model_metadata(
            diagnostic_fold0=FOLD0_DIAGNOSTIC,
            strict_oof=FOLD0_DIAGNOSTIC,
            production_metrics=None,
        )


def test_packaged_metadata_is_readable_by_the_entrypoint_validator(tmp_path):
    from ecup_matching.submission.predict_v7 import validate_v7_metadata

    report = build_submission_v7(
        model_dir=_model_dir(tmp_path),
        output_zip=tmp_path / "out.zip",
        diagnostic_fold0=FOLD0_DIAGNOSTIC,
    )
    with zipfile.ZipFile(report["zip"]) as archive:
        payload = json.loads(archive.read("model_v7_metadata.json"))
    validated = validate_v7_metadata(payload)
    assert validated["max_length"] == 256
    assert validated["inference_batch_size"] == 64


def test_archive_refuses_competition_data_and_credentials(tmp_path):
    model = _model_dir(tmp_path)
    (model / "leaked.parquet").write_bytes(b"\x00")
    with pytest.raises(ValueError, match="competition data"):
        build_submission_v7(model_dir=model, output_zip=tmp_path / "out.zip")

    (model / "leaked.parquet").unlink()
    (model / "hf_token").write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="credentials"):
        build_submission_v7(model_dir=model, output_zip=tmp_path / "out.zip")


def test_archive_rejects_a_model_directory_without_weights(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="safetensors"):
        build_submission_v7(model_dir=empty, output_zip=tmp_path / "out.zip")


def test_production_metrics_claiming_a_validation_score_are_rejected(tmp_path):
    metrics = tmp_path / "production-metrics.json"
    metrics.write_text(json.dumps({"validation_metric_reported": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="must not report a validation score"):
        build_submission_v7(
            model_dir=_model_dir(tmp_path),
            output_zip=tmp_path / "out.zip",
            production_metrics_path=metrics,
        )
