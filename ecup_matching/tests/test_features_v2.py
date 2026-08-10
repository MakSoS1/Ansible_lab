import json

import pandas as pd

from ecup_matching.ml.features_v2 import FEATURE_NAMES_V2, build_pair_features_v2


def _items():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": [
                "Apple iPhone 15 Pro 128GB Black",
                "Apple iPhone 15 Pro 256GB Black",
                "Apple iPhone 15 Pro 128GB Black",
            ],
            "attributes": [
                json.dumps({"brand": "Apple", "memory": "128 GB", "color": "black"}),
                json.dumps({"brand": "Apple", "memory": "256 GB", "color": "black"}),
                json.dumps({"brand": "Apple", "memory": "128 GB", "color": "black"}),
            ],
            "category": ["Electronics"] * 3,
        }
    )


def test_v2_features_include_2024_transfer_signals():
    pairs = pd.DataFrame({"id1": [1, 1], "id2": [2, 3]})
    importance = {"electronics": {"memory": 5.0, "color": 0.2, "brand": 1.0}}
    out = build_pair_features_v2(_items(), pairs, attribute_importance=importance)
    assert list(out.columns) == list(FEATURE_NAMES_V2)
    assert out.loc[0, "brand_match"] == 1.0
    assert out.loc[0, "brand_conflict"] == 0.0
    assert out.loc[0, "critical_conflict_count"] >= 1.0
    assert out.loc[0, "weighted_attr_conflict"] > out.loc[1, "weighted_attr_conflict"]
    assert out.loc[0, "hard_negative_score"] > out.loc[1, "hard_negative_score"]


def test_v2_features_are_symmetric():
    items = _items()
    importance = {"electronics": {"memory": 5.0, "color": 0.2, "brand": 1.0}}
    forward = build_pair_features_v2(items, pd.DataFrame({"id1": [1], "id2": [2]}), attribute_importance=importance)
    reverse = build_pair_features_v2(items, pd.DataFrame({"id1": [2], "id2": [1]}), attribute_importance=importance)
    pd.testing.assert_frame_equal(forward, reverse)
