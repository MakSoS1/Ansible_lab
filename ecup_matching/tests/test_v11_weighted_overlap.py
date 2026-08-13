import numpy as np
import pandas as pd

from ecup_matching.ml.v11_weighted_overlap import fit_weighted_overlap, predict_weighted_overlap


def test_weighted_overlap_learns_discriminative_shared_fragments():
    items = pd.DataFrame([
        {"id": 1, "name": "phone alpha 256 black", "attributes": {}, "category": "a"},
        {"id": 2, "name": "alpha phone 256 black", "attributes": {}, "category": "a"},
        {"id": 3, "name": "phone beta 128 white", "attributes": {}, "category": "a"},
        {"id": 4, "name": "beta case 128 white", "attributes": {}, "category": "a"},
        {"id": 5, "name": "phone gamma 512 blue", "attributes": {}, "category": "a"},
        {"id": 6, "name": "gamma phone 512 blue", "attributes": {}, "category": "a"},
    ])
    train = pd.DataFrame({"id1": [1, 1, 3, 3, 5, 5], "id2": [2, 4, 4, 2, 6, 4], "target": [1, 0, 1, 0, 1, 0], "category": ["a"] * 6})
    bundle = fit_weighted_overlap(items, train, n_features=4096, min_category_rows=4)
    score = predict_weighted_overlap(bundle, items, train[["id1", "id2", "category"]])
    assert np.isfinite(score).all()
    assert score[train.target.to_numpy() == 1].mean() > score[train.target.to_numpy() == 0].mean()
