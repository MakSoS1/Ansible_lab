"""The holdout is only meaningful if no item crosses the split.

If an endpoint appeared on both sides, the held Macro AP would partly measure
memorisation and would justify scaling decisions it cannot support, so
disjointness, group survival and the hard-target conversion are pinned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.v17_weak_holdout import split_weak_item_disjoint


def _frame(pairs, categories=None, targets=None):
    categories = categories or ["Электроника"] * len(pairs)
    targets = targets if targets is not None else [1.0, 0.0] * len(pairs)
    return pd.DataFrame(
        {
            "id1": [a for a, _ in pairs],
            "id2": [b for _, b in pairs],
            "target": targets[: len(pairs)],
            "category": categories,
            "_retrieval_anchor": [a for a, _ in pairs],
        }
    )


def test_no_item_appears_on_both_sides_of_the_split():
    rng = np.random.default_rng(17)
    pairs = [(int(a), int(a) + 500_000) for a in rng.integers(1, 4000, size=800)]
    frame = _frame(pairs, targets=list(rng.random(size=800)))

    train, held, report = split_weak_item_disjoint(
        frame, holdout_fraction=0.2, seed=2026
    )

    train_items = set(train["id1"]) | set(train["id2"])
    held_items = set(held["id1"]) | set(held["id2"])
    assert not (train_items & held_items)
    assert report["item_overlap"] == 0
    assert report["train_rows"] + report["held_rows"] == len(frame)


def test_a_chain_of_shared_items_travels_as_one_component():
    # 1-2, 2-3, 3-4 is one component; it must not be cut.
    frame = _frame([(1, 2), (2, 3), (3, 4), (90, 91)], targets=[1.0, 1.0, 0.0, 1.0])
    train, held, _ = split_weak_item_disjoint(frame, holdout_fraction=0.4, seed=7)

    chain = {1, 2, 3, 4}
    on_train = bool(chain & (set(train["id1"]) | set(train["id2"])))
    on_held = bool(chain & (set(held["id1"]) | set(held["id2"])))
    assert on_train != on_held


def test_retrieval_anchor_groups_are_never_split():
    frame = _frame(
        [(1, 10), (1, 11), (1, 12), (2, 20), (2, 21)],
        targets=[1.0, 0.0, 0.0, 1.0, 0.0],
    )
    train, held, _ = split_weak_item_disjoint(frame, holdout_fraction=0.3, seed=3)
    for anchor in (1, 2):
        in_train = int((train["_retrieval_anchor"] == anchor).sum())
        in_held = int((held["_retrieval_anchor"] == anchor).sum())
        assert in_train == 0 or in_held == 0


def test_held_targets_are_binarised_for_average_precision():
    frame = _frame([(1, 2), (3, 4), (5, 6), (7, 8)], targets=[0.9, 0.1, 0.51, 0.49])
    _, held, report = split_weak_item_disjoint(frame, holdout_fraction=0.45, seed=11)
    assert set(np.unique(held["target"])) <= {0, 1}
    assert 0.0 <= report["held_positive_rate"] <= 1.0


def test_degenerate_fractions_are_rejected():
    frame = _frame([(1, 2), (3, 4)])
    with pytest.raises(ValueError):
        split_weak_item_disjoint(frame, holdout_fraction=0.0, seed=1)
    with pytest.raises(ValueError):
        split_weak_item_disjoint(frame, holdout_fraction=0.8, seed=1)
