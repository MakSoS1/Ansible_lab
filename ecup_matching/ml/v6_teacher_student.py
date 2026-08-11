from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .v5_fixed_blend import percentile_rank
from .v5_meta_blend import rank_matrix


NON_TEACHER_SIGNAL_NAMES = (
    "weak",
    "sparse",
    "explicit",
    "contrastive",
    "typed_explicit",
)

DEFAULT_TEACHER_STUDENT_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "max_depth": 5,
    "min_samples_leaf": 100,
    "l2_regularization": 5.0,
    "early_stopping": False,
    "random_state": 20260811,
}


def _category_balanced_sample_weight(categories: np.ndarray) -> np.ndarray:
    cat = np.asarray(categories).astype(str)
    names, counts = np.unique(cat, return_counts=True)
    if len(names) == 0:
        raise ValueError("categories must not be empty")
    count_by_name = {name: int(count) for name, count in zip(names, counts, strict=True)}
    scale = len(cat) / len(names)
    return np.asarray([scale / count_by_name[value] for value in cat], dtype=np.float64)


def _ordered_non_teacher_scores(scores: Mapping[str, object]) -> dict[str, object]:
    missing = [name for name in NON_TEACHER_SIGNAL_NAMES if name not in scores]
    if missing:
        raise ValueError(f"missing non-teacher signals: {missing}")
    return {name: scores[name] for name in NON_TEACHER_SIGNAL_NAMES}


def _category_names(categories: np.ndarray) -> tuple[str, ...]:
    names = tuple(sorted(np.unique(np.asarray(categories).astype(str)).tolist()))
    if not names:
        raise ValueError("categories must not be empty")
    if len(names) > 255:
        raise ValueError("student categorical feature requires <=255 categories")
    return names


def _design_matrix(
    non_teacher_scores: Mapping[str, object],
    categories: np.ndarray,
    category_names: tuple[str, ...],
) -> np.ndarray:
    ranks = rank_matrix(_ordered_non_teacher_scores(non_teacher_scores))
    cat = np.asarray(categories).astype(str)
    if len(cat) != len(ranks):
        raise ValueError("categories must align with scores")
    code_by_name = {name: idx for idx, name in enumerate(category_names)}
    try:
        codes = np.asarray([code_by_name[value] for value in cat], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"unknown category: {exc.args[0]!r}") from exc
    return np.column_stack([ranks, codes])


def _new_model(**overrides) -> HistGradientBoostingRegressor:
    params = dict(DEFAULT_TEACHER_STUDENT_PARAMS)
    params.update(overrides)
    params["early_stopping"] = False
    return HistGradientBoostingRegressor(
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=False,
        random_state=int(params["random_state"]),
        categorical_features=[False, False, False, False, False, True],
    )


def crossfit_teacher_student(
    non_teacher_scores: Mapping[str, object],
    teacher_score,
    categories,
    folds,
    **overrides,
) -> dict[str, object]:
    teacher = np.asarray(teacher_score, dtype=np.float64)
    cat = np.asarray(categories).astype(str)
    fold_array = np.asarray(folds)
    if teacher.ndim != 1 or cat.ndim != 1 or fold_array.ndim != 1:
        raise ValueError("teacher_score, categories and folds must be one-dimensional")
    if not np.isfinite(teacher).all():
        raise ValueError("teacher_score contains non-finite values")
    if not (len(teacher) == len(cat) == len(fold_array)):
        raise ValueError("teacher_score, categories and folds must have equal lengths")
    names = _category_names(cat)
    design = _design_matrix(non_teacher_scores, cat, names)
    if len(design) != len(teacher):
        raise ValueError("non-teacher scores must align with teacher_score")

    unique_folds = sorted(np.unique(fold_array).tolist())
    if len(unique_folds) < 2:
        raise ValueError("cross-fitting requires at least two folds")
    oof = np.full(len(teacher), np.nan, dtype=np.float64)
    models: dict[int, HistGradientBoostingRegressor] = {}
    all_indices = np.arange(len(teacher), dtype=np.int64)
    for fold in unique_folds:
        valid_mask = fold_array == fold
        train_indices = all_indices[~valid_mask]
        valid_indices = all_indices[valid_mask]
        # Critical leakage safeguard: rank the teacher target using only the
        # outer-train partition. Held-fold teacher values are never read.
        teacher_target = percentile_rank(teacher[train_indices])
        model = _new_model(**overrides)
        model.fit(
            design[train_indices],
            teacher_target,
            sample_weight=_category_balanced_sample_weight(cat[train_indices]),
        )
        oof[valid_indices] = np.clip(model.predict(design[valid_indices]), 0.0, 1.0)
        models[int(fold)] = model
    if not np.isfinite(oof).all():
        raise RuntimeError("teacher student did not score every held row")
    return {
        "oof_score": oof,
        "fold_models": models,
        "category_names": names,
        "params": {**DEFAULT_TEACHER_STUDENT_PARAMS, **overrides, "early_stopping": False},
    }


def fit_teacher_student_full(
    non_teacher_scores: Mapping[str, object],
    teacher_score,
    categories,
    **overrides,
) -> dict[str, object]:
    teacher = np.asarray(teacher_score, dtype=np.float64)
    cat = np.asarray(categories).astype(str)
    if teacher.ndim != 1 or cat.ndim != 1 or len(teacher) != len(cat):
        raise ValueError("teacher_score and categories must be aligned one-dimensional vectors")
    if not np.isfinite(teacher).all():
        raise ValueError("teacher_score contains non-finite values")
    names = _category_names(cat)
    design = _design_matrix(non_teacher_scores, cat, names)
    model = _new_model(**overrides)
    model.fit(
        design,
        percentile_rank(teacher),
        sample_weight=_category_balanced_sample_weight(cat),
    )
    return {
        "model": model,
        "category_names": names,
        "params": {**DEFAULT_TEACHER_STUDENT_PARAMS, **overrides, "early_stopping": False},
    }


def predict_teacher_student(
    bundle: Mapping[str, object],
    non_teacher_scores: Mapping[str, object],
    categories,
) -> np.ndarray:
    category_names = tuple(str(value) for value in bundle["category_names"])
    design = _design_matrix(
        non_teacher_scores,
        np.asarray(categories).astype(str),
        category_names,
    )
    model = bundle["model"]
    return np.clip(np.asarray(model.predict(design), dtype=np.float64), 0.0, 1.0)
