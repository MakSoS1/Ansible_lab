from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

from . import run_v7_production as base
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .v10_student_contract import V10StudentConfig, validate_v10_student_config


EXPECTED_BASE_MODEL = "cointegrated/rubert-tiny2"
DEFAULT_TINY_REVISION = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"


def _validate_v10_config_from_v7_shape(config):
    validate_v10_student_config(
        V10StudentConfig(
            max_length=int(config.max_length),
            curriculum_rows=int(config.curriculum_rows),
            effective_batch_size=int(config.effective_batch_size),
            epochs=float(config.epochs),
            max_steps=None if config.max_steps is None else int(config.max_steps),
        )
    )
    return config


def normalize_v10_production_payload(
    payload: Mapping[str, Any],
    *,
    base_model_revision: str,
    inference_batch_size: int,
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.update(
        {
            "version": "v10-production-refit",
            "candidate": "rubert-tiny2-128-single-student",
            "base_model": EXPECTED_BASE_MODEL,
            "base_model_revision": str(base_model_revision),
            "runtime_architecture": "single-small-cross-encoder",
            "inference_batch_size": int(inference_batch_size),
            "validation_metric_reported": False,
            "gold_metric_opened": False,
            "gold_rows_scored": 0,
            "production_refit_is_validation": False,
        }
    )
    return normalized


def _argument_value(flag: str, default: str | None = None) -> str | None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


def _normalize_output(output_dir: Path) -> None:
    old_model = output_dir / "model_v7_teacher"
    new_model = output_dir / "model_v10_student"
    if old_model.is_dir():
        if new_model.exists():
            raise RuntimeError(f"v10 output already contains {new_model}")
        old_model.rename(new_model)
    if not new_model.is_dir():
        raise RuntimeError("v10 production refit did not produce a student model directory")

    metrics_path = output_dir / "production-metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("v10 production refit is missing production-metrics.json")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    revision = _argument_value("--base-model-revision", DEFAULT_TINY_REVISION)
    normalized = normalize_v10_production_payload(
        payload,
        base_model_revision=str(revision),
        inference_batch_size=512,
    )
    normalized["model_dir"] = "model_v10_student"
    metrics_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    original_manifest = base._build_immutable_manifest
    original_validator = base.validate_v7_teacher_config
    try:
        base._build_immutable_manifest = _load_immutable_manifest
        base.validate_v7_teacher_config = _validate_v10_config_from_v7_shape
        rc = base.main()
    finally:
        base._build_immutable_manifest = original_manifest
        base.validate_v7_teacher_config = original_validator
    if rc != 0:
        return rc
    output = _argument_value("--output-dir")
    if output is None:
        raise RuntimeError("v10 production wrapper requires --output-dir")
    _normalize_output(Path(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
