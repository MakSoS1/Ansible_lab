from __future__ import annotations

from dataclasses import dataclass

from .v7_teacher_contract import required_optimizer_steps


@dataclass(frozen=True)
class V10StudentConfig:
    max_length: int
    curriculum_rows: int
    effective_batch_size: int
    epochs: float
    max_steps: int | None = None


def validate_v10_student_config(config: V10StudentConfig) -> V10StudentConfig:
    """Validate training settings for the deliberately short v10 student.

    v7's >=256-token requirement remains untouched. v10 has the opposite
    runtime contract: short sequence length is an architectural requirement,
    not an optimization to be applied after validation.
    """
    if not 64 <= int(config.max_length) <= 160:
        raise ValueError("v10 max_length must be within [64, 160]")
    needed = required_optimizer_steps(config)
    if config.max_steps is not None:
        if int(config.max_steps) <= 0:
            raise ValueError("max_steps must be positive when specified")
        if int(config.max_steps) < needed:
            raise ValueError(
                f"v10 optimizer steps are truncated: max_steps={config.max_steps}, required={needed}"
            )
    return config


__all__ = ["V10StudentConfig", "validate_v10_student_config"]
