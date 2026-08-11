from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


_REQUIRED_SOURCES: tuple[str, ...] = ("category", "weak", "sparse", "explicit")


def _finite_1d(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def percentile_rank(values) -> np.ndarray:
    """Return average percentile ranks in [0, 1] without using labels."""
    array = _finite_1d(values, name="values")
    if len(array) == 0:
        raise ValueError("values must not be empty")
    if len(array) == 1:
        return np.array([0.5], dtype=np.float64)
    raw_rank = pd.Series(array).rank(method="average").to_numpy(dtype=np.float64)
    # Map rank 1..N to 0..1. Ties receive the average mapped rank.
    return (raw_rank - 1.0) / float(len(array) - 1)


def fixed_blend_candidates(
    scores: Mapping[str, object],
    *,
    contrastive_cosine=None,
) -> dict[str, np.ndarray]:
    """Build predeclared target-free fusion candidates from held-out scores.

    No target labels, learned coefficients, category-specific tuning or metric
    feedback enter this function. Source names are semantic and make results
    independent of input mapping order.
    """
    missing = [name for name in _REQUIRED_SOURCES if name not in scores]
    if missing:
        raise ValueError(f"missing required score sources: {missing}")

    arrays = {name: _finite_1d(scores[name], name=name) for name in _REQUIRED_SOURCES}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all score sources must have equal length")
    row_count = next(iter(lengths))
    if row_count == 0:
        raise ValueError("score sources must not be empty")

    clipped = {
        name: np.clip(values, 1e-6, 1.0 - 1e-6)
        for name, values in arrays.items()
    }
    ranks = {name: percentile_rank(values) for name, values in arrays.items()}

    result: dict[str, np.ndarray] = {
        "prob_mean_4": np.mean(
            np.vstack([clipped[name] for name in _REQUIRED_SOURCES]), axis=0
        ),
        "rank_mean_3": np.mean(
            np.vstack([ranks[name] for name in ("weak", "sparse", "explicit")]), axis=0
        ),
        "rank_mean_4": np.mean(
            np.vstack([ranks[name] for name in _REQUIRED_SOURCES]), axis=0
        ),
    }

    if contrastive_cosine is not None:
        cosine = _finite_1d(contrastive_cosine, name="contrastive_cosine")
        if len(cosine) != row_count:
            raise ValueError("contrastive_cosine and score sources must have equal length")
        result["rank_mean_5"] = np.mean(
            np.vstack([ranks[name] for name in _REQUIRED_SOURCES] + [percentile_rank(cosine)]),
            axis=0,
        )

    return result
