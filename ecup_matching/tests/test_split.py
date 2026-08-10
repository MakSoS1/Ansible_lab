import pandas as pd

from ecup_matching.ml.split import component_split


def test_component_split_is_item_disjoint_complete_and_deterministic():
    pairs = pd.DataFrame(
        {
            "id1": [1, 2, 10, 20, 21, 30],
            "id2": [2, 3, 11, 21, 22, 31],
            "target": [1, 0, 1, 0, 1, 0],
            "category": ["a", "a", "a", "b", "b", "b"],
        }
    )

    train_idx, valid_idx = component_split(pairs, valid_fraction=0.34, seed=2026)
    train_idx2, valid_idx2 = component_split(pairs, valid_fraction=0.34, seed=2026)

    assert train_idx.tolist() == train_idx2.tolist()
    assert valid_idx.tolist() == valid_idx2.tolist()
    assert sorted(train_idx.tolist() + valid_idx.tolist()) == list(range(len(pairs)))
    assert set(train_idx).isdisjoint(set(valid_idx))
    assert len(valid_idx) > 0
    assert len(train_idx) > 0

    train_items = set(pairs.iloc[train_idx]["id1"]) | set(pairs.iloc[train_idx]["id2"])
    valid_items = set(pairs.iloc[valid_idx]["id1"]) | set(pairs.iloc[valid_idx]["id2"])
    assert train_items.isdisjoint(valid_items)
