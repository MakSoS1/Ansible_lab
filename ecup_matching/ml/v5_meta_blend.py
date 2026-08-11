from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

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
