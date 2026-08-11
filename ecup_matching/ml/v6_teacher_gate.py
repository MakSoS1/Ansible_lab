from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .v5_fixed_blend import percentile_rank
from .v5_meta_blend import SIX_SIGNAL_NAMES


GATE_COVERAGES = (0.25, 0.40, 0.55, 0.70, 0.85, 0.95)
NON_TEACHER_SIGNALS = (
    "weak",
    "sparse",
    "explicit",
    "contrastive",
    "typed_explicit",
)


def _finite_vector(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _ranked_non_teacher(signals: Mapping[str, object]) -> dict[str, np.ndarray]:
    ranked: dict[str, np.ndarray] = {}
    length: int | None = None
    for name in NON_TEACHER_SIGNALS:
        if name not in signals:
            raise ValueError(f"missing signal: {name}")
        values = _finite_vector(signals[name], name=name)
        if length is None:
            length = len(values)
        elif len(values) != length:
            raise ValueError("signal lengths do not match")
        ranked[name] = percentile_rank(values)
    return ranked


def disagreement_gate_mask(
    non_teacher_signals: Mapping[str, object],
    categories: Sequence[object],
    *,
    coverage: float,
) -> np.ndarray:
    if not np.isfinite(coverage) or coverage <= 0.0 or coverage > 1.0:
        raise ValueError("coverage must be in (0, 1]")
    ranked = _ranked_non_teacher(non_teacher_signals)
    matrix = np.column_stack([ranked[name] for name in NON_TEACHER_SIGNALS])
    category_array = np.asarray(categories, dtype=object)
    if category_array.ndim != 1 or len(category_array) != len(matrix):
        raise ValueError("categories must be one-dimensional and align with scores")
    category_strings = category_array.astype(str)
    disagreement = np.std(matrix, axis=1, dtype=np.float64)
    mask = np.zeros(len(matrix), dtype=bool)
    for category in sorted(set(category_strings.tolist())):
        indices = np.flatnonzero(category_strings == category)
        if len(indices) == 0:
            continue
        take = min(len(indices), max(1, int(np.ceil(float(coverage) * len(indices)))))
        local = disagreement[indices]
        order = np.argsort(-local, kind="mergesort")[:take]
        mask[indices[order]] = True
    return mask


def assemble_partial_teacher_signal(
    non_teacher_signals: Mapping[str, object],
    categories: Sequence[object],
    *,
    coverage: float,
    selected_teacher_scores,
) -> tuple[np.ndarray, np.ndarray]:
    ranked_non_teacher = _ranked_non_teacher(non_teacher_signals)
    mask = disagreement_gate_mask(ranked_non_teacher, categories, coverage=coverage)
    selected = _finite_vector(selected_teacher_scores, name="selected_teacher_scores")
    expected_selected = int(mask.sum())
    if len(selected) != expected_selected:
        raise ValueError(
            "selected_teacher_scores length does not match gate selection: "
            f"expected {expected_selected}, got {len(selected)}"
        )
    surrogate = np.mean(
        np.column_stack([ranked_non_teacher[name] for name in NON_TEACHER_SIGNALS]),
        axis=1,
        dtype=np.float64,
    )
    mixed_teacher = surrogate.copy()
    if expected_selected:
        mixed_teacher[mask] = percentile_rank(selected)
    return mixed_teacher, mask


def build_teacher_gated_scores(
    signals: Mapping[str, object],
    categories: Sequence[object],
    *,
    coverage: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    ranked_non_teacher = _ranked_non_teacher(signals)
    teacher = _finite_vector(signals.get("teacher"), name="teacher")
    expected_length = len(next(iter(ranked_non_teacher.values())))
    if len(teacher) != expected_length:
        raise ValueError("teacher length does not match non-teacher signals")
    mask = disagreement_gate_mask(ranked_non_teacher, categories, coverage=coverage)
    mixed_teacher, partial_mask = assemble_partial_teacher_signal(
        signals,
        categories,
        coverage=coverage,
        selected_teacher_scores=teacher[mask],
    )
    if not np.array_equal(mask, partial_mask):
        raise RuntimeError("teacher gate mask changed during assembly")

    result: dict[str, np.ndarray] = {}
    for name in SIX_SIGNAL_NAMES:
        if name == "teacher":
            result[name] = mixed_teacher
        else:
            if name not in signals:
                raise ValueError(f"missing signal: {name}")
            values = _finite_vector(signals[name], name=name)
            if len(values) != expected_length:
                raise ValueError("signal lengths do not match")
            result[name] = values
    return result, mask
