from __future__ import annotations

from typing import Iterable

import numpy as np

from .metrics import macro_average_precision


def _as_scores(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def apply_blend(structured, neural, *, alpha: float) -> np.ndarray:
    """Blend structured and neural probabilities with neural weight ``alpha``."""
    s = _as_scores(structured, name="structured")
    n = _as_scores(neural, name="neural")
    if len(s) != len(n):
        raise ValueError("structured and neural scores must have the same length")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    return np.clip((1.0 - alpha) * s + alpha * n, 0.0, 1.0)


def select_global_blend(
    structured,
    neural,
    target,
    categories,
    *,
    alphas: Iterable[float] | None = None,
) -> dict[str, object]:
    """Choose one global blend coefficient by macro AP on supplied rows.

    Ties intentionally prefer the simpler endpoint/lowest neural weight. This
    keeps the search conservative and avoids per-category validation overfit.
    """
    s = _as_scores(structured, name="structured")
    n = _as_scores(neural, name="neural")
    y = _as_scores(target, name="target")
    c = np.asarray(categories)
    if c.ndim != 1:
        raise ValueError("categories must be one-dimensional")
    if not (len(s) == len(n) == len(y) == len(c)):
        raise ValueError("blend inputs must have the same length")
    if len(s) == 0:
        raise ValueError("blend inputs must not be empty")

    grid_values = np.asarray(
        list(alphas) if alphas is not None else np.linspace(0.0, 1.0, 11),
        dtype=np.float64,
    )
    if grid_values.ndim != 1 or not len(grid_values):
        raise ValueError("alphas must contain at least one value")
    if not np.isfinite(grid_values).all() or ((grid_values < 0) | (grid_values > 1)).any():
        raise ValueError("all alphas must be finite and in [0,1]")
    grid_values = np.unique(grid_values)

    results: list[dict[str, float]] = []
    best_alpha: float | None = None
    best_score = -np.inf
    best_per_category: dict[str, float] | None = None
    for alpha in grid_values:
        score = apply_blend(s, n, alpha=float(alpha))
        macro, per_category = macro_average_precision(y, score, c)
        results.append({"alpha": float(alpha), "macro_average_precision": float(macro)})
        if macro > best_score + 1e-12:
            best_alpha = float(alpha)
            best_score = float(macro)
            best_per_category = per_category

    assert best_alpha is not None and best_per_category is not None
    return {
        "alpha": best_alpha,
        "macro_average_precision": best_score,
        "per_category_ap": best_per_category,
        "grid": results,
    }
