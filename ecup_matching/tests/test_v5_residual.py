import numpy as np
import pandas as pd

from ecup_matching.ml.v5_residual import (
    ResidualRanker,
    apply_residual,
    clipped_logit,
)


def test_zero_residual_preserves_base_scores_exactly_up_to_roundoff():
    base = np.array([1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-6], dtype=float)
    corrected = apply_residual(base, np.zeros_like(base), residual_strength=1.0)
    assert np.allclose(corrected, base, rtol=0.0, atol=1e-12)
    assert np.array_equal(np.argsort(corrected), np.argsort(base))


def test_clipped_logit_is_finite_for_boundary_probabilities():
    values = clipped_logit(np.array([0.0, 1.0, 0.5]))
    assert np.isfinite(values).all()
    assert values[0] < 0 < values[1]
    assert values[2] == 0.0


def test_residual_ranker_can_learn_small_correction_without_replacing_anchor():
    x = pd.DataFrame(
        {
            "conflict": [0.0, 0.0, 1.0, 1.0] * 20,
            "category": ["a", "b", "a", "b"] * 20,
        }
    )
    base = np.array([0.8, 0.7, 0.8, 0.7] * 20, dtype=float)
    target = np.array([1, 1, 0, 0] * 20, dtype=int)

    ranker = ResidualRanker(residual_strength=0.35, l2_regularization=8.0, seed=2026)
    ranker.fit(x, target, base)
    corrected = ranker.predict_proba(x, base)

    assert corrected.shape == base.shape
    assert np.isfinite(corrected).all()
    assert ((corrected > 0.0) & (corrected < 1.0)).all()
    assert corrected[x["conflict"].to_numpy() == 1].mean() < base[x["conflict"].to_numpy() == 1].mean()
    # With strong L2 and residual_strength < 1, correction remains bounded.
    assert np.max(np.abs(clipped_logit(corrected) - clipped_logit(base))) < 2.5
