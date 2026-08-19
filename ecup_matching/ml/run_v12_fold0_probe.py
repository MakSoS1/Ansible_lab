from __future__ import annotations

from . import v7_teacher_contract as _contract


def _validate_v12_config(config: _contract.V7TeacherConfig) -> _contract.V7TeacherConfig:
    """V12 keeps v7 optimizer/leakage checks but tests a shorter GPU context."""
    if config.max_length < 160:
        raise ValueError("v12 max_length must be at least 160")
    needed = _contract.required_optimizer_steps(config)
    if config.max_steps is not None:
        if config.max_steps <= 0:
            raise ValueError("max_steps must be positive when specified")
        if int(config.max_steps) < needed:
            raise ValueError(
                f"v12 optimizer steps are truncated: max_steps={config.max_steps}, required={needed}"
            )
    return config


# run_v7_fold0_probe imports the validator by value. Install the isolated v12
# validator before importing that module so all split/filtering/training logic
# remains byte-for-byte the proven v7 path.
_contract.validate_v7_teacher_config = _validate_v12_config

from .run_v7_fold0_probe import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
