from pathlib import Path

import numpy as np
import pandas as pd

from ecup_matching.ml.v11_fastlex import FEATURE_NAMES, build_fast_pair_features
from ecup_matching.ml.v11_sparse import SparseConfig, sparse_pair_scores
from ecup_matching.ml.v11_stack import crossfit_hgb_scores


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


def test_sparse_semantic_channel_is_deterministic_and_prefers_near_duplicate():
    pairs = pd.DataFrame({"id1": [1, 1], "id2": [2, 3]})
    cfg = SparseConfig(n_features=32768)
    first = sparse_pair_scores(_items(), pairs, config=cfg)
    second = sparse_pair_scores(_items(), pairs, config=cfg)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
    assert np.isfinite(first).all()
    assert first[0] > first[1]


def test_crossfit_stack_does_not_use_held_fold_labels_for_that_fold():
    n = 80
    x = np.linspace(0.0, 1.0, n)
    features = pd.DataFrame({"category": np.where(np.arange(n) % 2, "a", "b"), "lex": x, "sparse": x**2})
    folds = np.repeat([0, 1], n // 2)
    target = (x > 0.55).astype(np.int8)
    base = crossfit_hgb_scores(features, target, folds, min_local_rows=10)
    changed = target.copy(); changed[folds == 0] = 1 - changed[folds == 0]
    perturbed = crossfit_hgb_scores(features, changed, folds, min_local_rows=10)
    np.testing.assert_allclose(base[folds == 0], perturbed[folds == 0], rtol=0, atol=0)
    assert np.isfinite(base).all()
