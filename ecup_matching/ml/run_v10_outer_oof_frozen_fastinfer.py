from __future__ import annotations

from . import run_v7_outer_oof as base
from . import run_v7_outer_oof_frozen_fastinfer as frozen_fast
from .v10_student_contract import V10StudentConfig, validate_v10_student_config


def _validate_v10_config_from_v7_shape(config):
    """Adapt the shared OOF driver's config shape without weakening v7 rules."""
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


def main() -> int:
    original_validator = base.validate_v7_teacher_config
    try:
        base.validate_v7_teacher_config = _validate_v10_config_from_v7_shape
        return frozen_fast.main()
    finally:
        base.validate_v7_teacher_config = original_validator


if __name__ == "__main__":
    raise SystemExit(main())
