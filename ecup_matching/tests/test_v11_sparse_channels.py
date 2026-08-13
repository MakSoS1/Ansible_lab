import numpy as np
import pandas as pd
from ecup_matching.ml.v11_sparse_channels import build_sparse_pair_features


def test_sparse_channels_prefer_near_duplicate():
    items = pd.DataFrame([
        {"id": 1, "name": "Apple iPhone 15 Pro 256GB", "attributes": {"color": "black", "memory": "256 gb"}, "category": "phones"},
        {"id": 2, "name": "iphone 15 pro 256 gb apple", "attributes": {"color": "black", "memory": "256GB"}, "category": "phones"},
        {"id": 3, "name": "Apple iPhone 14 128GB", "attributes": {"color": "white", "memory": "128 gb"}, "category": "phones"},
    ])
    pairs = pd.DataFrame({"id1": [1, 1], "id2": [2, 3]})
    frame = build_sparse_pair_features(items, pairs, max_name_features=2048, max_char_features=4096, max_attr_features=1024)
    assert list(frame["category"]) == ["phones", "phones"]
    assert np.isfinite(frame.drop(columns=["category"]).to_numpy(float)).all()
    for col in ("name_word_cosine", "name_char_cosine", "attr_cosine"):
        assert frame.iloc[0][col] > frame.iloc[1][col]
