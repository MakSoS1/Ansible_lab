from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .v5_meta_blend import rank_matrix


DEFAULT_HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 160,
    "max_leaf_nodes": 15,
    "max_depth": 3,
    "min_samples_leaf": 200,
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


def _design_matrix(
    ranks: np.ndarray,
    categories: np.ndarray,
    category_names: tuple[str, ...],
) -> np.ndarray:
    cat = np.asarray(categories).astype(str)
    code_by_name = {name: idx for idx, name in enumerate(category_names)}
    try:
        codes = np.asarray([code_by_name[value] for value in cat], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"unknown category: {exc.args[0]!r}") from exc
    return np.column_stack([np.asarray(ranks, dtype=np.float64), codes])


def _fit_model(
    design: np.ndarray,
    target: np.ndarray,
    categories: np.ndarray,
    train_indices: np.ndarray,
    *,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    max_depth: int,
    min_samples_leaf: int,
    l2_regularization: float,
    random_state: int,
) -> HistGradientBoostingClassifier:
    y_train = target[train_indices]
    if set(np.unique(y_train).tolist()) != {0, 1}:
        raise ValueError("each HGB training partition must contain both target classes")
    sample_weight = _category_balanced_sample_weight(categories[train_indices])
    model = HistGradientBoostingClassifier(
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        max_leaf_nodes=int(max_leaf_nodes),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        l2_regularization=float(l2_regularization),
        early_stopping=False,
        random_state=int(random_state),
        categorical_features=[False, False, False, False, False, False, True],
    )
    model.fit(design[train_indices], y_train, sample_weight=sample_weight)
    return model


def crossfit_fixed_hgb_stack(
    scores: Mapping[str, object],
    target,
    categories,
    folds,
    *,
    learning_rate: float = 0.05,
    max_iter: int = 160,
    max_leaf_nodes: int = 15,
    max_depth: int = 3,
    min_samples_leaf: int = 200,
    l2_regularization: float = 5.0,
    random_state: int = 20260811,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, object]:
    """Score each immutable outer fold with a fixed nonlinear meta model.

    The rank transform and category vocabulary are target-free. Hyperparameters
    are fixed before evaluation, and each held fold is excluded completely from
    the corresponding HGB fit.
    """
    ranks = rank_matrix(scores)
    y = np.asarray(target, dtype=np.int8)
    cat = np.asarray(categories).astype(str)
    fold_array = np.asarray(folds)
    n_rows = len(ranks)
    if y.ndim != 1 or cat.ndim != 1 or fold_array.ndim != 1:
        raise ValueError("target, categories and folds must be one-dimensional")
    if not (len(y) == len(cat) == len(fold_array) == n_rows):
        raise ValueError("scores, target, categories and folds must have equal lengths")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("target must contain both binary classes")
    if not np.isfinite(fold_array.astype(np.float64)).all():
        raise ValueError("folds must be finite")

    unique_folds = sorted(np.unique(fold_array).tolist())
    if len(unique_folds) < 2:
        raise ValueError("cross-fitting requires at least two folds")
    category_names = tuple(sorted(np.unique(cat).tolist()))
    if len(category_names) > 255:
        raise ValueError("HGB categorical feature requires <=255 category values")
    design = _design_matrix(ranks, cat, category_names)
    all_indices = np.arange(n_rows, dtype=np.int64)
    oof = np.full(n_rows, np.nan, dtype=np.float64)
    fold_models: dict[int, HistGradientBoostingClassifier] = {}

    for done, fold in enumerate(unique_folds, start=1):
        valid = fold_array == fold
        train = ~valid
        train_indices = all_indices[train]
        valid_indices = all_indices[valid]
        model = _fit_model(
            design,
            y,
            cat,
            train_indices,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )
        oof[valid_indices] = model.predict_proba(design[valid_indices])[:, 1]
        fold_models[int(fold)] = model
        if progress is not None:
            progress(done, len(unique_folds), int(fold))

    if not np.isfinite(oof).all():
        raise RuntimeError("HGB cross-fit did not score every row exactly once")
    return {
        "oof_score": oof,
        "fold_models": fold_models,
        "rank_matrix": ranks,
        "category_names": category_names,
        "design_matrix": design,
        "params": {
            "learning_rate": float(learning_rate),
            "max_iter": int(max_iter),
            "max_leaf_nodes": int(max_leaf_nodes),
            "max_depth": int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            "l2_regularization": float(l2_regularization),
            "early_stopping": False,
            "random_state": int(random_state),
        },
    }


def fit_fixed_hgb_full(
    scores: Mapping[str, object],
    target,
    categories,
    **overrides,
) -> dict[str, object]:
    """Fit the exact frozen HGB architecture on all development rows for production."""
    ranks = rank_matrix(scores)
    y = np.asarray(target, dtype=np.int8)
    cat = np.asarray(categories).astype(str)
    category_names = tuple(sorted(np.unique(cat).tolist()))
    design = _design_matrix(ranks, cat, category_names)
    params = dict(DEFAULT_HGB_PARAMS)
    params.update(overrides)
    model = _fit_model(
        design,
        y,
        cat,
        np.arange(len(y), dtype=np.int64),
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        random_state=params["random_state"],
    )
    return {"model": model, "category_names": category_names, "params": params}
