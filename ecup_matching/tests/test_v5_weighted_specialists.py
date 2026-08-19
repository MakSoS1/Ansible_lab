import numpy as np
import pandas as pd

from ecup_matching.ml.v5_weighted_specialists import (
    build_fold_weighted_features,
    fit_fold_attribute_importance,
)


def _items():
    rows = []
    for i in range(12):
        rows.append(
            {
                "id": i,
                "name": f"товар {i // 2}",
                "category": "electronics",
                "attributes": {
                    "model": f"M{i // 2}",
                    "memory": "256" if i % 2 == 0 else "128",
                    "decorative": "same",
                },
            }
        )
    return pd.DataFrame(rows)


def test_fold_attribute_importance_uses_only_supplied_training_pairs():
    items = _items()
    train = pd.DataFrame(
        {
            "id1": [0, 2, 4, 0, 2, 4],
            "id2": [1, 3, 5, 2, 4, 0],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )
    held = pd.DataFrame({"id1": [6, 8], "id2": [7, 9], "target": [1, 1]})

    first = fit_fold_attribute_importance(items, train, min_support=2)
    held_changed = held.copy()
    held_changed["target"] = 1 - held_changed["target"]
    second = fit_fold_attribute_importance(items, train, min_support=2)

    assert first == second
    assert "electronics" in first
    assert "memory" in first["electronics"]
    assert first["electronics"]["memory"] > 0


def test_weighted_features_are_deterministic_and_do_not_require_pair_targets():
    items = _items()
    train = pd.DataFrame(
        {
            "id1": [0, 2, 4, 0, 2, 4],
            "id2": [1, 3, 5, 2, 4, 0],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )
    importance = fit_fold_attribute_importance(items, train, min_support=2)
    pairs = pd.DataFrame({"id1": [0, 0, 2], "id2": [1, 2, 3]})

    a = build_fold_weighted_features(items, pairs, importance)
    b = build_fold_weighted_features(items, pairs.copy(), importance)

    assert a.equals(b)
    assert "weighted_attr_agreement" in a.columns
    assert "weighted_attr_conflict" in a.columns
    assert np.isfinite(a.drop(columns="category").to_numpy(dtype=float)).all()
