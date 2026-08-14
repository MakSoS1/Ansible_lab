from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from .metrics import macro_average_precision
from .v5_fixed_blend import percentile_rank


SIX_SIGNAL_NAMES: tuple[str, ...] = (
    "weak",
    "sparse",
    "explicit",
    "contrastive",
    "teacher",
    "typed_explicit",
)


def _finite_1d(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_six_signal_scores(scores: Mapping[str, object]) -> dict[str, np.ndarray]:
    missing = [name for name in SIX_SIGNAL_NAMES if name not in scores]
    if missing:
        raise ValueError(f"missing required score sources: {missing}")
    arrays = {name: _finite_1d(scores[name], name=name) for name in SIX_SIGNAL_NAMES}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all score sources must have equal length")
    return arrays


def rank_matrix(scores: Mapping[str, object]) -> np.ndarray:
    """Convert the six heterogeneous scores to the same label-free rank scale."""
    arrays = _validated_six_signal_scores(scores)
    return np.column_stack([percentile_rank(arrays[name]) for name in SIX_SIGNAL_NAMES])


def _validate_fit_inputs(matrix, target, categories) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target)
    cat = np.asarray(categories).astype(str)
    if x.ndim != 2 or x.shape[1] != len(SIX_SIGNAL_NAMES):
        raise ValueError(f"matrix must have shape (n_rows, {len(SIX_SIGNAL_NAMES)})")
    if len(x) == 0:
        raise ValueError("matrix must not be empty")
    if not np.isfinite(x).all():
        raise ValueError("matrix must contain only finite values")
    if y.ndim != 1 or cat.ndim != 1 or not (len(x) == len(y) == len(cat)):
        raise ValueError("matrix, target and categories must have aligned row counts")
    unique_target = set(np.unique(y).tolist())
    if not unique_target.issubset({0, 1}) or len(unique_target) < 2:
        raise ValueError(f"target must contain both binary classes; observed={sorted(unique_target)}")
    return x, y.astype(np.int8, copy=False), cat


def _simplex(values) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != len(SIX_SIGNAL_NAMES):
        raise ValueError(f"weights must have length {len(SIX_SIGNAL_NAMES)}")
    if not np.isfinite(weights).all() or np.any(weights < -1e-12):
        raise ValueError("weights must be finite and nonnegative")
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    return weights / total


def _objective(x: np.ndarray, y: np.ndarray, categories: np.ndarray, weights: np.ndarray) -> float:
    score = x @ weights
    macro, _ = macro_average_precision(y, score, categories)
    return float(macro)


def fit_simplex_weights(
    matrix,
    target,
    categories,
    *,
    initial_weights: Sequence[float] | None = None,
    step_schedule: Sequence[float] = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0, 1.0 / 96.0),
    max_passes: int = 6,
    min_improvement: float = 1e-9,
) -> np.ndarray:
    """Fit deterministic nonnegative global weights directly for Macro AP.

    Search is a bounded pairwise mass-transfer coordinate ascent on the simplex.
    No held-out labels are touched by this function; outer cross-fitting is handled
    by :func:`crossfit_global_simplex_blend`.
    """
    x, y, cat = _validate_fit_inputs(matrix, target, categories)
    if max_passes < 1:
        raise ValueError("max_passes must be >= 1")
    steps = tuple(float(step) for step in step_schedule)
    if not steps or any((not np.isfinite(step)) or step <= 0.0 or step >= 1.0 for step in steps):
        raise ValueError("step_schedule must contain finite values strictly between 0 and 1")

    if initial_weights is None:
        weights = np.full(len(SIX_SIGNAL_NAMES), 1.0 / len(SIX_SIGNAL_NAMES), dtype=np.float64)
    else:
        weights = _simplex(initial_weights)
    best_score = _objective(x, y, cat, weights)

    for step in steps:
        for _ in range(max_passes):
            pass_weights = weights
            pass_score = best_score
            candidate_weights: np.ndarray | None = None
            candidate_score = pass_score

            for source in range(len(weights)):
                if weights[source] + 1e-15 < step:
                    continue
                for destination in range(len(weights)):
                    if source == destination:
                        continue
                    trial = weights.copy()
                    trial[source] -= step
                    trial[destination] += step
                    score = _objective(x, y, cat, trial)
                    if score > candidate_score + min_improvement:
                        candidate_score = score
                        candidate_weights = trial

            if candidate_weights is None:
                weights = pass_weights
                best_score = pass_score
                break
            weights = candidate_weights
            best_score = candidate_score

    return _simplex(weights)


def crossfit_global_simplex_blend(
    scores: Mapping[str, object],
    target,
    categories,
    folds,
    *,
    step_schedule: Sequence[float] = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0, 1.0 / 96.0),
    max_passes: int = 6,
    min_improvement: float = 1e-9,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> dict[str, object]:
    """Produce strict outer-fold OOF scores with fold-local target fitting only."""
    arrays = _validated_six_signal_scores(scores)
    n_rows = len(next(iter(arrays.values())))
    y = np.asarray(target)
    cat = np.asarray(categories).astype(str)
    fold_array = np.asarray(folds)
    if y.ndim != 1 or cat.ndim != 1 or fold_array.ndim != 1:
        raise ValueError("target, categories and folds must be one-dimensional")
    if not (len(y) == len(cat) == len(fold_array) == n_rows):
        raise ValueError("scores, target, categories and folds must have equal lengths")
    if not np.isfinite(fold_array.astype(np.float64)).all():
        raise ValueError("folds must be finite")

    x = rank_matrix(arrays)
    unique_folds = sorted(np.unique(fold_array).tolist())
    if len(unique_folds) < 2:
        raise ValueError("cross-fitting requires at least two folds")

    oof = np.full(n_rows, np.nan, dtype=np.float64)
    fold_weights: dict[int, np.ndarray] = {}
    for done, fold in enumerate(unique_folds, start=1):
        valid = fold_array == fold
        train = ~valid
        if not valid.any() or not train.any():
            raise ValueError(f"fold {fold!r} must have both train and validation rows")
        weights = fit_simplex_weights(
            x[train],
            y[train],
            cat[train],
            step_schedule=step_schedule,
            max_passes=max_passes,
            min_improvement=min_improvement,
        )
        oof[valid] = x[valid] @ weights
        fold_weights[int(fold)] = weights
        if progress is not None:
            progress(done, len(unique_folds), int(train.sum()), int(valid.sum()))

    if not np.isfinite(oof).all():
        raise RuntimeError("cross-fit did not produce exactly one finite score for every row")
    return {
        "oof_score": oof,
        "fold_weights": fold_weights,
        "rank_matrix": x,
    }


def _category_interaction_design(
    rank_values: np.ndarray,
    categories: np.ndarray,
    category_names: Sequence[str],
) -> sparse.csr_matrix:
    """Global rank terms plus category-specific rank adjustments.

    Category vocabulary and rank transforms are target-free, so constructing the
    design over the full development batch does not expose held-out labels.
    """
    x = np.asarray(rank_values, dtype=np.float64) - 0.5
    cat = np.asarray(categories).astype(str)
    names = tuple(str(value) for value in category_names)
    code_by_name = {name: idx for idx, name in enumerate(names)}
    try:
        codes = np.asarray([code_by_name[value] for value in cat], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"unknown category in design matrix: {exc.args[0]!r}") from exc

    n_rows, n_signals = x.shape
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), n_signals)
    columns = np.repeat(codes.astype(np.int64), n_signals) * n_signals + np.tile(
        np.arange(n_signals, dtype=np.int64), n_rows
    )
    interactions = sparse.csr_matrix(
        (x.reshape(-1), (rows, columns)),
        shape=(n_rows, len(names) * n_signals),
        dtype=np.float64,
    )
    return sparse.hstack([sparse.csr_matrix(x), interactions], format="csr")


def _category_balanced_sample_weight(categories: np.ndarray) -> np.ndarray:
    cat = np.asarray(categories).astype(str)
    names, counts = np.unique(cat, return_counts=True)
    count_by_name = {name: int(count) for name, count in zip(names, counts, strict=True)}
    n_categories = len(names)
    if n_categories == 0:
        raise ValueError("categories must not be empty")
    scale = len(cat) / n_categories
    return np.asarray([scale / count_by_name[value] for value in cat], dtype=np.float64)


def _fit_logistic(
    design: sparse.csr_matrix,
    target: np.ndarray,
    categories: np.ndarray,
    indices: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
) -> LogisticRegression:
    y = np.asarray(target, dtype=np.int8)[indices]
    if len(np.unique(y)) != 2:
        raise ValueError("each logistic training partition must contain both target classes")
    weights = _category_balanced_sample_weight(np.asarray(categories)[indices])
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        max_iter=int(max_iter),
        tol=1e-6,
    )
    model.fit(design[indices], y, sample_weight=weights)
    return model


def crossfit_nested_category_logistic(
    scores: Mapping[str, object],
    target,
    categories,
    folds,
    *,
    c_grid: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0),
    max_iter: int = 250,
    progress: Callable[[int, int, int, float, float], None] | None = None,
) -> dict[str, object]:
    """Fully nested category-aware logistic stack.

    For every outer fold, regularization is selected by an inner OOF over only
    the remaining original folds. The outer fold's labels therefore cannot
    affect either its selected hyperparameter or its fitted coefficients.
    """
    arrays = _validated_six_signal_scores(scores)
    n_rows = len(next(iter(arrays.values())))
    y = np.asarray(target, dtype=np.int8)
    cat = np.asarray(categories).astype(str)
    fold_array = np.asarray(folds)
    if y.ndim != 1 or cat.ndim != 1 or fold_array.ndim != 1:
        raise ValueError("target, categories and folds must be one-dimensional")
    if not (len(y) == len(cat) == len(fold_array) == n_rows):
        raise ValueError("scores, target, categories and folds must have equal lengths")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("target must contain both binary classes")
    if not np.isfinite(fold_array.astype(np.float64)).all():
        raise ValueError("folds must be finite")

    unique_folds = sorted(np.unique(fold_array).tolist())
    if len(unique_folds) < 3:
        raise ValueError("nested logistic cross-fitting requires at least three outer folds")
    c_values = tuple(sorted({float(value) for value in c_grid}))
    if not c_values or any((not np.isfinite(value)) or value <= 0.0 for value in c_values):
        raise ValueError("c_grid must contain positive finite values")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    ranks = rank_matrix(arrays)
    category_names = tuple(sorted(np.unique(cat).tolist()))
    design = _category_interaction_design(ranks, cat, category_names)
    all_indices = np.arange(n_rows, dtype=np.int64)
    oof = np.full(n_rows, np.nan, dtype=np.float64)
    selected_c: dict[int, float] = {}
    inner_scores: dict[int, dict[float, float]] = {}

    for done, outer_fold in enumerate(unique_folds, start=1):
        outer_valid = fold_array == outer_fold
        outer_train = ~outer_valid
        train_indices = all_indices[outer_train]
        valid_indices = all_indices[outer_valid]
        inner_fold_ids = [fold for fold in unique_folds if fold != outer_fold]
        candidate_scores: dict[float, float] = {}

        for c_value in c_values:
            inner_oof = np.full(n_rows, np.nan, dtype=np.float64)
            for inner_fold in inner_fold_ids:
                inner_valid = outer_train & (fold_array == inner_fold)
                inner_fit = outer_train & (fold_array != inner_fold)
                fit_indices = all_indices[inner_fit]
                held_indices = all_indices[inner_valid]
                model = _fit_logistic(
                    design,
                    y,
                    cat,
                    fit_indices,
                    c_value=c_value,
                    max_iter=max_iter,
                )
                inner_oof[held_indices] = model.decision_function(design[held_indices])

            train_score = inner_oof[train_indices]
            if not np.isfinite(train_score).all():
                raise RuntimeError(f"inner OOF incomplete for outer fold {outer_fold}")
            macro, _ = macro_average_precision(y[train_indices], train_score, cat[train_indices])
            candidate_scores[c_value] = float(macro)

        best_score = max(candidate_scores.values())
        chosen_c = min(
            value for value, score in candidate_scores.items() if abs(score - best_score) <= 1e-12
        )
        final_model = _fit_logistic(
            design,
            y,
            cat,
            train_indices,
            c_value=chosen_c,
            max_iter=max_iter,
        )
        oof[valid_indices] = final_model.decision_function(design[valid_indices])
        selected_c[int(outer_fold)] = float(chosen_c)
        inner_scores[int(outer_fold)] = candidate_scores
        if progress is not None:
            progress(done, len(unique_folds), int(outer_fold), float(chosen_c), float(best_score))

    if not np.isfinite(oof).all():
        raise RuntimeError("nested category logistic did not score every row exactly once")
    return {
        "oof_score": oof,
        "selected_c": selected_c,
        "inner_macro_ap_by_c": inner_scores,
        "rank_matrix": ranks,
        "category_names": category_names,
    }
