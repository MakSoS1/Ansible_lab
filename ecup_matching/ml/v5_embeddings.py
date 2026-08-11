from __future__ import annotations

import numpy as np


EMBEDDING_PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "embedding_cosine",
    "embedding_mean_abs_diff",
    "embedding_l2",
    "embedding_max_abs_diff",
    "embedding_mean_product",
    "embedding_min_product",
    "embedding_max_product",
)


def build_embedding_pair_features(embedding_a, embedding_b) -> np.ndarray:
    """Return compact symmetric pair statistics for two embedding batches."""
    a = np.asarray(embedding_a, dtype=np.float64)
    b = np.asarray(embedding_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("embeddings must be 2D arrays")
    if a.shape != b.shape:
        raise ValueError("embedding batches must have identical shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("embeddings contain NaN or infinity")
    if a.shape[1] == 0:
        raise ValueError("embedding dimension must be positive")

    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    denom = norm_a * norm_b
    dot = np.sum(a * b, axis=1)
    cosine = np.zeros(len(a), dtype=np.float64)
    nonzero = denom > 0
    cosine[nonzero] = dot[nonzero] / denom[nonzero]
    both_zero = (norm_a == 0) & (norm_b == 0)
    cosine[both_zero] = 1.0
    cosine = np.clip(cosine, -1.0, 1.0)

    abs_diff = np.abs(a - b)
    product = a * b
    return np.column_stack(
        [
            cosine,
            abs_diff.mean(axis=1),
            np.linalg.norm(a - b, axis=1),
            abs_diff.max(axis=1),
            product.mean(axis=1),
            product.min(axis=1),
            product.max(axis=1),
        ]
    )
