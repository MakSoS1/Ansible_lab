import numpy as np
import pandas as pd

from ecup_matching.ml.v11_sparse import SparseConfig, sparse_pair_scores


def test_sparse_prefers_similar_product():
    items = pd.DataFrame([
        {"id": 1, "name": "Samsung Galaxy S24 Ultra 256GB", "attributes": {}, "category": "phones"},
        {"id": 2, "name": "galaxy s24 ultra samsung 256 gb", "attributes": {}, "category": "phones"},
        {"id": 3, "name": "Apple Watch Series 9 45mm", "attributes": {}, "category": "wearables"},
    ])
    pairs = pd.DataFrame({"id1": [1, 1], "id2": [2, 3]})
    cfg = SparseConfig(n_features=32768)
    a = sparse_pair_scores(items, pairs, config=cfg)
    b = sparse_pair_scores(items, pairs, config=cfg)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    assert np.isfinite(a).all()
    assert a[0] > a[1]


def test_sparse_identical_text_is_near_one():
    items = pd.DataFrame([
        {"id": 1, "name": "robot vacuum x10", "attributes": {}, "category": "vacuum"},
        {"id": 2, "name": "robot vacuum x10", "attributes": {}, "category": "vacuum"},
    ])
    pairs = pd.DataFrame({"id1": [1], "id2": [2]})
    assert sparse_pair_scores(items, pairs, config=SparseConfig(n_features=4096))[0] > 0.999
