from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .v5_meta_blend import (
    SIX_SIGNAL_NAMES,
    fit_simplex_weights,
    rank_matrix,
)


def _validated_problem(scores, target, categories) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rank_matrix(scores)
    y = np.asarray(target, dtype=np.int8)
    cat = np.asarray(categories).astype(str)
    if y.ndim != 1 or cat.ndim != 1:
        raise ValueError("target and categories must be one-dimensional")
    if not (len(y) == len(cat) == len(x)):
        raise ValueError("scores, target and categories must have equal lengths")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("target must contain both binary classes")
    return x, y, cat


def _shrunk_category_weights(
    x: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
    global_weights: np.ndarray,
    *,
    prior_strength: float,
    step_schedule: Sequence[float],
    max_passes: int,
    min_improvement: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    result: dict[str, np.ndarray] = {}
    support_by_category: dict[str, int] = {}
    for category in sorted(np.unique(categories).tolist()):
        local = categories == category
        support = int(local.sum())
        support_by_category[category] = support
        if support == 0 or len(np.unique(y[local])) < 2:
            local_weights = global_weights.copy()
        else:
            local_weights = fit_simplex_weights(
                x[local],
                y[local],
                categories[local],
                initial_weights=global_weights,
                step_schedule=step_schedule,
                max_passes=max_passes,
                min_improvement=min_improvement,
            )
        shrunk = (
            support * local_weights + float(prior_strength) * global_weights
        ) / (support + float(prior_strength))
        shrunk = np.maximum(shrunk, 0.0)
        total = float(shrunk.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise RuntimeError(f"invalid shrunk weights for category {category!r}")
        result[category] = shrunk / total
    return result, support_by_category


def fit_category_shrunk_full(
    scores: Mapping[str, object],
    target,
    categories,
    *,
    prior_strength: float = 8000.0,
    step_schedule: Sequence[float] = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0),
    max_passes: int = 4,
    min_improvement: float = 1e-9,
) -> dict[str, object]:
    """Fit the selected fixed category-shrunk ensemble on all development OOF rows.

    This is a production refit after model selection. Its in-sample score must not
    be used as validation; the corresponding honest selection score is obtained
    from :func:`crossfit_category_shrunk_simplex`.
    """
    prior_strength = float(prior_strength)
    if not np.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive and finite")
    x, y, cat = _validated_problem(scores, target, categories)
    global_weights = fit_simplex_weights(
        x,
        y,
        cat,
        step_schedule=step_schedule,
        max_passes=max_passes,
        min_improvement=min_improvement,
    )
    category_weights, category_support = _shrunk_category_weights(
        x,
        y,
        cat,
        global_weights,
        prior_strength=prior_strength,
        step_schedule=step_schedule,
        max_passes=max_passes,
        min_improvement=min_improvement,
    )
    return {
        "global_weights": global_weights,
        "category_weights": category_weights,
        "category_support": category_support,
        "rank_matrix": x,
        "signal_names": SIX_SIGNAL_NAMES,
        "prior_strength": prior_strength,
    }


def crossfit_category_shrunk_simplex(
    scores: Mapping[str, object],
    target,
    categories,
    folds,
    *,
    prior_strength: float = 8000.0,
    step_schedule: Sequence[float] = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0),
    max_passes: int = 4,
    min_improvement: float = 1e-9,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, object]:
    """Outer-cross-fitted global + category-local simplex with fixed shrinkage.

    The shrinkage prior is a fixed hyperparameter. For every outer fold all
    global and category-local weights are fit exclusively on the other folds.
    """
    x, y, cat = _validated_problem(scores, target, categories)
    fold_array = np.asarray(folds)
    if fold_array.ndim != 1 or len(fold_array) != len(x):
        raise ValueError("folds must be one-dimensional and aligned with scores")
    if not np.isfinite(fold_array.astype(np.float64)).all():
        raise ValueError("folds must be finite")
    prior_strength = float(prior_strength)
    if not np.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive and finite")

    unique_folds = sorted(np.unique(fold_array).tolist())
    if len(unique_folds) < 2:
        raise ValueError("cross-fitting requires at least two folds")
    category_names = tuple(sorted(np.unique(cat).tolist()))

    oof = np.full(len(x), np.nan, dtype=np.float64)
    global_weights: dict[int, np.ndarray] = {}
    category_weights: dict[int, dict[str, np.ndarray]] = {}
    category_support: dict[int, dict[str, int]] = {}

    for done, fold in enumerate(unique_folds, start=1):
        train = fold_array != fold
        valid = ~train
        train_indices = np.flatnonzero(train)
        valid_indices = np.flatnonzero(valid)
        global_w = fit_simplex_weights(
            x[train_indices],
            y[train_indices],
            cat[train_indices],
            step_schedule=step_schedule,
            max_passes=max_passes,
            min_improvement=min_improvement,
        )
        global_weights[int(fold)] = global_w
        local_weight_map, local_support_map = _shrunk_category_weights(
            x[train_indices],
            y[train_indices],
            cat[train_indices],
            global_w,
            prior_strength=prior_strength,
            step_schedule=step_schedule,
            max_passes=max_passes,
            min_improvement=min_improvement,
        )
        category_weights[int(fold)] = local_weight_map
        category_support[int(fold)] = local_support_map

        for category in category_names:
            local_valid = valid & (cat == category)
            if not local_valid.any():
                continue
            if category not in local_weight_map:
                raise RuntimeError(f"outer-train partition missing category {category!r}")
            positions = np.flatnonzero(local_valid)
            oof[positions] = x[positions] @ local_weight_map[category]

        if not np.isfinite(oof[valid_indices]).all():
            raise RuntimeError(f"category-shrunk fold {fold} did not score every held-out row")
        if progress is not None:
            progress(done, len(unique_folds), int(fold))

    if not np.isfinite(oof).all():
        raise RuntimeError("category-shrunk cross-fit did not score every row exactly once")
    return {
        "oof_score": oof,
        "global_weights": global_weights,
        "category_weights": category_weights,
        "category_support": category_support,
        "rank_matrix": x,
        "signal_names": SIX_SIGNAL_NAMES,
        "prior_strength": prior_strength,
    }
