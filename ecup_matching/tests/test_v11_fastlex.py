from pathlib import Path

import numpy as np
import pandas as pd

from ecup_matching.ml.v11_fastlex import FEATURE_NAMES, build_fast_pair_features


def _items():
    return pd.DataFrame([
        {"id": 1, "name": "Apple iPhone 15 Pro 256GB", "attributes": {"color": "black", "memory": "256 gb"}, "category": "phones"},
        {"id": 2, "name": "iphone 15 pro 256 gb apple", "attributes": {"color": "black", "memory": "256GB"}, "category": "phones"},
        {"id": 3, "name": "Apple iPhone 14 128GB", "attributes": {"color": "white", "memory": "128 gb"}, "category": "phones"},
    ])


def test_fastlex_preserves_pair_order_and_has_finite_numeric_features():
    pairs = pd.DataFrame({"id1": [1, 1], "id2": [3, 2]})
    frame = build_fast_pair_features(_items(), pairs)
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert len(frame) == 2
    numeric = frame.drop(columns=["category"])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
    assert frame.iloc[1]["name_token_jaccard"] > frame.iloc[0]["name_token_jaccard"]


def test_fastlex_captures_codes_numbers_and_attributes_without_edit_distance():
    pairs = pd.DataFrame({"id1": [1, 1], "id2": [2, 3]})
    frame = build_fast_pair_features(_items(), pairs)
    near, far = frame.iloc[0], frame.iloc[1]
    assert near["number_jaccard"] > far["number_jaccard"]
    assert near["attr_value_agreement"] > far["attr_value_agreement"]
    assert near["quantity_conflict"] <= far["quantity_conflict"]
    source = Path("ecup_matching/ml/v11_fastlex.py").read_text(encoding="utf-8")
    assert "SequenceMatcher" not in source
    assert "difflib" not in source
