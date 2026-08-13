from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class SparseConfig:
    n_features: int = 65536
    batch_size: int = 50000
    ngram_range: tuple[int, int] = (1, 2)


def sparse_pair_scores(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    config: SparseConfig = SparseConfig(),
) -> np.ndarray:
    if not {"id", "name"}.issubset(items.columns):
        raise ValueError("items must contain id and name")
    if not {"id1", "id2"}.issubset(pairs.columns):
        raise ValueError("pairs must contain id1 and id2")
    if items["id"].duplicated().any():
        raise ValueError("items.id must be unique")

    names = items["name"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=config.ngram_range,
        max_features=int(config.n_features),
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(names).tocsr()
    row_of = {item_id: row for row, item_id in enumerate(items["id"].tolist())}
    try:
        left = np.fromiter((row_of[x] for x in pairs["id1"]), dtype=np.int64, count=len(pairs))
        right = np.fromiter((row_of[x] for x in pairs["id2"]), dtype=np.int64, count=len(pairs))
    except KeyError as exc:
        raise KeyError(f"pair references missing item: {exc.args[0]!r}") from exc

    scores = np.empty(len(pairs), dtype=np.float32)
    batch = max(1, int(config.batch_size))
    for start in range(0, len(pairs), batch):
        stop = min(start + batch, len(pairs))
        product = matrix[left[start:stop]].multiply(matrix[right[start:stop]])
        scores[start:stop] = np.asarray(product.sum(axis=1)).ravel().astype(np.float32, copy=False)
    np.clip(scores, 0.0, 1.0, out=scores)
    return scores


__all__ = ["SparseConfig", "sparse_pair_scores"]
