"""Carve an item-disjoint evaluation slice out of the weak corpus.

Human fold-0 has been a poor leaderboard predictor: v13b scored the best local
fold-0 of any candidate (`0.7086611386`) and came third of four on the public
leaderboard, while v14 scored below it locally and first externally. The
population audit explains why that axis is narrow — human labelling covers
711,304 items, `5.31%` of the 13,397,761-item universe, and the weak pool
covers `92.44%` of it with zero shared ids.

So the weak pool is the only population we hold that is anywhere near the size
of the one we are scored on, and nothing has ever been measured on it. This
module splits the prepared weak frame into train and held slices whose item
sets are disjoint, so a model cannot score the held slice by having memorised
an endpoint.

Splitting on connected components rather than on rows is what makes that
guarantee hold: an item reached through any edge travels with its whole
component. Retrieval-anchor groups survive automatically, because every row of
an anchor shares `id1` and therefore lands in one component.

The held labels are the weak targets, so this measures agreement with the LLM
annotator, not with human truth. It answers "does this model generalise to
unseen items in the population the test set lives in", which is a different
and currently unmeasured question.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _component_roots(id1: np.ndarray, id2: np.ndarray) -> dict[int, int]:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(id1.tolist(), id2.tolist()):
        a, b = int(a), int(b)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {int(node): find(int(node)) for node in parent}


def _stable_order_key(root: int, seed: int) -> bytes:
    return hashlib.sha256(f"{int(seed)}\0{int(root)}".encode("utf-8")).digest()


def split_weak_item_disjoint(
    weak: pd.DataFrame,
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Split into (train, held) with no shared item id.

    Components are taken in a stable pseudo-random order until the requested
    row fraction is reached, rather than thresholded on a hash. A threshold
    only approximates the fraction and can empty a side outright when there are
    few components; ordering hits the requested size closely and always leaves
    both sides populated whenever more than one component exists. A component
    is never partially taken, so the disjointness guarantee is unconditional.
    """
    if not 0.0 < float(holdout_fraction) < 0.5:
        raise ValueError("holdout_fraction must be inside (0, 0.5)")
    for column in ("id1", "id2", "target", "category"):
        if column not in weak.columns:
            raise ValueError(f"weak frame must contain {column!r}")
    if weak.empty:
        raise ValueError("weak frame must not be empty")

    id1 = weak["id1"].to_numpy().astype(np.int64)
    id2 = weak["id2"].to_numpy().astype(np.int64)
    roots = _component_roots(id1, id2)
    root_series = pd.Series(roots)

    left_root = weak["id1"].map(root_series).to_numpy()
    rows_by_root = pd.Series(left_root).value_counts().to_dict()
    ordered = sorted(rows_by_root, key=lambda root: _stable_order_key(int(root), seed))

    target_rows = float(holdout_fraction) * len(weak)
    held_roots: set[int] = set()
    taken = 0
    for root in ordered:
        if len(held_roots) == len(ordered) - 1:
            break  # never empty the training side
        if taken >= target_rows:
            break
        held_roots.add(int(root))
        taken += int(rows_by_root[root])

    held_mask = np.fromiter(
        (int(root) in held_roots for root in left_root), dtype=bool, count=len(weak)
    )
    train = weak.loc[~held_mask].reset_index(drop=True)
    held = weak.loc[held_mask].reset_index(drop=True)
    if train.empty or held.empty:
        raise RuntimeError("weak holdout split produced an empty side")

    train_items = set(train["id1"].tolist()) | set(train["id2"].tolist())
    held_items = set(held["id1"].tolist()) | set(held["id2"].tolist())
    overlap = train_items & held_items
    if overlap:
        raise RuntimeError(f"weak holdout split leaked {len(overlap)} items")

    held = held.copy()
    held["target"] = (held["target"].astype(float) >= 0.5).astype(np.int8)

    report: dict[str, object] = {
        "components": int(len(set(roots.values()))),
        "train_rows": int(len(train)),
        "held_rows": int(len(held)),
        "requested_holdout_fraction": float(holdout_fraction),
        "realised_holdout_fraction": float(len(held) / max(len(weak), 1)),
        "train_items": int(len(train_items)),
        "held_items": int(len(held_items)),
        "item_overlap": 0,
        "held_categories": int(held["category"].astype(str).nunique()),
        "held_positive_rate": float(held["target"].mean()),
    }
    return train, held, report
