from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .v6_teacher_gate import disagreement_gate_mask
from .v6_teacher_student import crossfit_teacher_student


HYBRID_COVERAGES = (0.25, 0.40, 0.55, 0.70, 0.85)


def empirical_cdf_from_train(train_values, query_values) -> np.ndarray:
    train = np.asarray(train_values, dtype=np.float64)
    query = np.asarray(query_values, dtype=np.float64)
    if train.ndim != 1 or len(train) == 0 or not np.isfinite(train).all():
        raise ValueError("train_values must be a finite non-empty vector")
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("query_values must be a finite vector")
    ordered = np.sort(train, kind="mergesort")
    ranks = np.searchsorted(ordered, query, side="right")
    return np.clip(ranks.astype(np.float64) / float(len(ordered)), 0.0, 1.0)


def build_crossfit_hybrid_teacher(
    non_teacher_signals: Mapping[str, object],
    teacher_score,
    categories: Sequence[object],
    folds,
    *,
    coverage: float,
) -> dict[str, object]:
    if coverage not in HYBRID_COVERAGES:
        raise ValueError(f"coverage must be one of {HYBRID_COVERAGES}")
    teacher = np.asarray(teacher_score, dtype=np.float64)
    cat = np.asarray(categories).astype(str)
    fold_array = np.asarray(folds)
    if teacher.ndim != 1 or cat.ndim != 1 or fold_array.ndim != 1:
        raise ValueError("teacher_score, categories and folds must be one-dimensional")
    if not np.isfinite(teacher).all():
        raise ValueError("teacher_score contains non-finite values")
    if not (len(teacher) == len(cat) == len(fold_array)):
        raise ValueError("teacher_score, categories and folds must have equal lengths")

    selected = disagreement_gate_mask(
        non_teacher_signals,
        cat,
        coverage=coverage,
    )
    student = crossfit_teacher_student(
        non_teacher_signals,
        teacher,
        cat,
        fold_array,
    )
    hybrid = np.asarray(student["oof_score"], dtype=np.float64).copy()
    unique_folds = sorted(np.unique(fold_array).tolist())
    for fold in unique_folds:
        held = fold_array == fold
        train = ~held
        selected_held = held & selected
        if selected_held.any():
            hybrid[selected_held] = empirical_cdf_from_train(
                teacher[train], teacher[selected_held]
            )
    if not np.isfinite(hybrid).all():
        raise RuntimeError("hybrid teacher did not score every row")
    return {
        "hybrid_teacher_score": np.clip(hybrid, 0.0, 1.0),
        "teacher_selected": selected,
        "student_teacher_score": np.asarray(student["oof_score"], dtype=np.float64),
        "student_fold_models": student["fold_models"],
        "student_category_names": student["category_names"],
    }
