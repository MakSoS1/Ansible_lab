from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    sd = arr.std()
    return (arr - arr.mean()) / (sd if sd > 1e-12 else 1.0)


def acquisition_scores(
    candidates: np.ndarray,
    training: np.ndarray,
    predicted_npv: np.ndarray,
    uncertainty: np.ndarray,
    *,
    value_weight: float = 1.0,
    uncertainty_weight: float = 0.5,
    novelty_weight: float = 0.25,
) -> np.ndarray:
    cand = np.asarray(candidates, dtype=float)
    train = np.asarray(training, dtype=float)
    if cand.ndim != 2 or train.ndim != 2 or cand.shape[1] != train.shape[1]:
        raise ValueError("candidate and training controls must be 2D with the same feature count")
    novelty = cdist(cand, train).min(axis=1) if len(train) else np.ones(len(cand))
    return (
        value_weight * _zscore(predicted_npv)
        + uncertainty_weight * _zscore(uncertainty)
        + novelty_weight * _zscore(novelty)
    )


def select_for_opm(scores: np.ndarray, *, budget: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if budget <= 0:
        return np.array([], dtype=int)
    return np.argsort(values)[-min(budget, len(values)):][::-1]
