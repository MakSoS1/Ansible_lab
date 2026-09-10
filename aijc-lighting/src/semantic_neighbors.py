from __future__ import annotations

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def fuse_normalized_views(views: list[np.ndarray]) -> np.ndarray:
    """Concatenate independently normalized embedding views and normalize again."""
    if not views:
        raise ValueError("at least one embedding view is required")
    n = len(views[0])
    if any(len(view) != n for view in views):
        raise ValueError("all views must contain the same number of rows")
    fused = np.concatenate([_l2_normalize(view) for view in views], axis=1)
    return _l2_normalize(fused)


def cosine_knn_probabilities(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    query_features: np.ndarray,
    *,
    k: int = 5,
    temperature: float = 0.07,
    exclude_self: bool = False,
    n_classes: int = 3,
) -> np.ndarray:
    """Temperature-weighted cosine kNN class probabilities."""
    ref = _l2_normalize(reference_features)
    query = _l2_normalize(query_features)
    labels = np.asarray(reference_labels, dtype=int)
    if len(ref) != len(labels):
        raise ValueError("reference feature/label length mismatch")
    if k < 1:
        raise ValueError("k must be positive")

    similarities = query @ ref.T
    if exclude_self:
        if len(query) != len(ref):
            raise ValueError("exclude_self requires equal query/reference row counts")
        np.fill_diagonal(similarities, -np.inf)

    kk = min(k, len(ref) - (1 if exclude_self else 0))
    idx = np.argpartition(-similarities, kth=max(kk - 1, 0), axis=1)[:, :kk]
    row = np.arange(len(query))[:, None]
    sims = similarities[row, idx]
    scaled = (sims - np.max(sims, axis=1, keepdims=True)) / max(float(temperature), 1e-6)
    weights = np.exp(scaled)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    probs = np.zeros((len(query), n_classes), dtype=np.float32)
    for c in range(n_classes):
        probs[:, c] = np.sum(weights * (labels[idx] == c), axis=1)
    return probs


def nearest_neighbor_margin(similarities: np.ndarray) -> np.ndarray:
    """Return top-1 minus top-2 similarity for each query."""
    similarities = np.asarray(similarities, dtype=np.float32)
    if similarities.ndim != 2 or similarities.shape[1] < 2:
        raise ValueError("similarities must be a matrix with at least two columns")
    top2 = np.partition(similarities, kth=similarities.shape[1] - 2, axis=1)[:, -2:]
    top2.sort(axis=1)
    return top2[:, 1] - top2[:, 0]
