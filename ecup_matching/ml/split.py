from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[object, object] = {}
        self.rank: dict[object, int] = {}

    def add(self, x: object) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: object) -> object:
        self.add(x)
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: object, b: object) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def component_split(
    matches: pd.DataFrame,
    valid_fraction: float = 0.2,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Split pair rows by connected item components so item IDs never leak across folds."""
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    required = {"id1", "id2"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches is missing required columns: {sorted(missing)}")
    if len(matches) < 2:
        raise ValueError("at least two pair rows are required for a train/validation split")

    uf = _UnionFind()
    for id1, id2 in matches[["id1", "id2"]].itertuples(index=False, name=None):
        uf.union(id1, id2)

    rows_by_component: dict[object, list[int]] = defaultdict(list)
    for row_idx, id1 in enumerate(matches["id1"].tolist()):
        rows_by_component[uf.find(id1)].append(row_idx)

    components = list(rows_by_component.items())
    if len(components) < 2:
        raise ValueError(
            "all pairs belong to one connected item component; an item-disjoint split is impossible"
        )

    rng = np.random.default_rng(seed)
    tie_break = {root: float(rng.random()) for root, _ in components}
    components.sort(key=lambda kv: (-len(kv[1]), tie_break[kv[0]]))

    target_valid = max(1, int(round(len(matches) * valid_fraction)))
    valid_components: set[object] = set()
    valid_rows = 0

    # Greedy bin packing: add a component when doing so moves validation size
    # closer to the requested row target. Keep at least one component for train.
    for pos, (root, rows) in enumerate(components):
        remaining_components = len(components) - pos - 1
        candidate = valid_rows + len(rows)
        improves = abs(candidate - target_valid) <= abs(valid_rows - target_valid)
        if improves and remaining_components >= 1:
            valid_components.add(root)
            valid_rows = candidate

    if not valid_components:
        # Pick the smallest component as validation to preserve as much training data as possible.
        root, rows = min(components, key=lambda kv: len(kv[1]))
        valid_components.add(root)
        valid_rows = len(rows)

    if len(valid_components) == len(components):
        # Move the largest validation component back to train.
        root, _ = max(components, key=lambda kv: len(kv[1]))
        valid_components.remove(root)

    valid_idx: list[int] = []
    train_idx: list[int] = []
    for root, rows in rows_by_component.items():
        (valid_idx if root in valid_components else train_idx).extend(rows)

    train = np.asarray(sorted(train_idx), dtype=np.int64)
    valid = np.asarray(sorted(valid_idx), dtype=np.int64)

    if len(train) == 0 or len(valid) == 0:
        raise RuntimeError("component split produced an empty fold")

    return train, valid
