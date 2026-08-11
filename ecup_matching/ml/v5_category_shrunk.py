from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .v5_meta_blend import (
    SIX_SIGNAL_NAMES,
    fit_simplex_weights,
    rank_matrix,
)


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
    y = np.asarray(target, dtype=np.int8)
    cat = np.asarray(categories).astype(str)
    fold_array = np.asarray(folds)
    x = rank_matrix(scores)
    n_rows = len(x)
    if y.ndim != 1 or cat.ndim != 1 or fold_array.ndim != 1:
        raise ValueError("target, categories and folds must be one-dimensional")
    if not (len(y) == len(cat) == len(fold_array) == n_rows):
        raise ValueError("scores, target, categories and folds must have equal lengths")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("target must contain both binary classes")
    if not np.isfinite(fold_array.astype(np.float64)).all():
        raise ValueError("folds must be finite")
    prior_strength = float(prior_strength)
    if not np.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive and finite")

    unique_folds = sorted(np.unique(fold_array).tolist())
    if len(unique_folds) < 2:
        raise ValueError("cross-fitting requires at least two folds")
    category_names = tuple(sorted(np.unique(cat).tolist()))

    oof = np.full(n_rows, np.nan, dtype=np.float64)
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
        category_weights[int(fold)] = {}
        category_support[int(fold)] = {}

        for category in category_names:
            local_train = train & (cat == category)
            local_valid = valid & (cat == category)
            support = int(local_train.sum())
            category_support[int(fold)][category] = support
            if support == 0:
                shrunk = global_w.copy()
            else:
                local_y = y[local_train]
                if len(np.unique(local_y)) < 2:
                    local_w = global_w.copy()
                else:
                    local_w = fit_simplex_weights(
                        x[local_train],
                        local_y,
                        cat[local_train],
                        initial_weights=global_w,
                        step_schedule=step_schedule,
                        max_passes=max_passes,
                        min_improvement=min_improvement,
                    )
                shrunk = (
                    support * local_w + prior_strength * global_w
                ) / (support + prior_strength)
                shrunk = np.maximum(shrunk, 0.0)
                shrunk = shrunk / shrunk.sum()
            category_weights[int(fold)][category] = shrunk
            if local_valid.any():
                positions = np.flatnonzero(local_valid)
                oof[positions] = x[positions] @ shrunk

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
