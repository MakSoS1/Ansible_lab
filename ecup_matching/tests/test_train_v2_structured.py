import numpy as np
import pandas as pd

from ecup_matching.ml.train_v2_structured import (
    candidate_sample_weights,
    fit_estimator,
    prefilter_weak_candidates,
)


def test_prefilter_weak_candidates_excludes_validation_items_and_mid_confidence():
    weak = pd.DataFrame(
        {
            "id1": [1, 3, 5, 7],
            "id2": [2, 4, 6, 8],
            "target": [0.99, 0.50, 0.10, 0.75],
        }
    )
    out = prefilter_weak_candidates(weak, validation_item_ids={2}, max_presample_rows=10, seed=2026)
    assert set(out[["id1", "id2"]].itertuples(index=False, name=None)) == {(5, 6), (7, 8)}
    assert np.allclose(out["weak_weight"].to_numpy(float), [0.6, 0.3])


def test_candidate_sample_weights_human_dominates_weak_and_hard_negatives_get_more_weight():
    categories = pd.Series(["a", "a", "a", "a"])
    source = pd.Series(["human", "weak", "human", "weak"])
    targets = np.array([1, 1, 0, 0], dtype=np.int8)
    weak_weight = np.array([1.0, 0.3, 1.0, 0.3], dtype=float)
    hard_score = np.array([0.0, 0.0, 0.8, 0.8], dtype=float)
    plain = candidate_sample_weights(categories, source, targets, weak_weight, hard_score, hard_negative_boost=0.0)
    hard = candidate_sample_weights(categories, source, targets, weak_weight, hard_score, hard_negative_boost=3.0)
    assert plain[0] > plain[1]
    # Global normalization is allowed; the semantic contract is that hard
    # negatives gain weight relative to corresponding positive examples.
    assert hard[2] / hard[0] > plain[2] / plain[0]
    assert hard[3] / hard[1] > plain[3] / plain[1]
    # Human/weak relative weighting is preserved by normalization.
    assert np.isclose(hard[0] / hard[1], plain[0] / plain[1])


def test_fit_estimator_supports_predict_proba_on_mixed_numeric_and_category_features():
    x = pd.DataFrame(
        {
            "category": ["a", "a", "b", "b"] * 8,
            "similarity": [0.9, 0.1, 0.8, 0.2] * 8,
            "conflict": [0.0, 1.0, 0.0, 1.0] * 8,
        }
    )
    y = np.array([1, 0, 1, 0] * 8, dtype=np.int8)
    model = fit_estimator(x, y, np.ones(len(y)), seed=2026)
    p = model.predict_proba(x)[:, 1]
    assert p.shape == (len(x),)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()
