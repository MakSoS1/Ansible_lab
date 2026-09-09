from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_splits(
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    dummy = np.zeros((len(y), 1), dtype=np.float32)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(dummy, y)
    else:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError('groups and y must have the same length')
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(dummy, y, groups)
