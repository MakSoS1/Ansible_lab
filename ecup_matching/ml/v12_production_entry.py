from __future__ import annotations

from . import v7_teacher_contract as contract


def validate_v12(config: contract.V7TeacherConfig) -> contract.V7TeacherConfig:
    if config.max_length < 160:
        raise ValueError("v12 max_length must be at least 160")
    needed = contract.required_optimizer_steps(config)
    if config.max_steps is not None and int(config.max_steps) < needed:
        raise ValueError(
            f"v12 optimizer steps are truncated: max_steps={config.max_steps}, required={needed}"
        )
    return config


contract.validate_v7_teacher_config = validate_v12

from .run_v7_production import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
