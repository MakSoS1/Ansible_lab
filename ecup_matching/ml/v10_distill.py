from __future__ import annotations

import numpy as np


def _as_finite_1d(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def blend_hard_teacher_targets(
    hard_targets,
    teacher_targets=None,
    *,
    teacher_weight: float,
) -> np.ndarray:
    """Build BCE targets while keeping hard-label-only validation explicit.

    This helper does not decide whether a teacher signal is leakage-safe. The
    caller must only pass teacher targets produced under the validation protocol
    appropriate to that run. Passing ``None`` with zero teacher weight is the
    canonical hard-label-only path used when fold-safe teacher targets are not
    available.
    """

    weight = float(teacher_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("teacher_weight must be finite and within [0, 1]")

    hard = _as_finite_1d(hard_targets, name="hard_targets")
    if not np.isin(hard, np.asarray([0.0, 1.0])).all():
        raise ValueError("hard_targets must be binary 0/1 labels")

    if teacher_targets is None:
        if weight != 0.0:
            raise ValueError("teacher_targets are required when teacher_weight is non-zero")
        return hard.copy()

    teacher = _as_finite_1d(teacher_targets, name="teacher_targets")
    if len(teacher) != len(hard):
        raise ValueError("teacher_targets length must match hard_targets length")
    if ((teacher < 0.0) | (teacher > 1.0)).any():
        raise ValueError("teacher_targets must lie within [0, 1]")

    return (1.0 - weight) * hard + weight * teacher


__all__ = ["blend_hard_teacher_targets"]
