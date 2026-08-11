import numpy as np
import pandas as pd

from ecup_matching.ml.v5_category_specialists import fit_predict_category_specialists


def test_category_specialists_predict_every_validation_row_in_original_order():
    train_x = pd.DataFrame(
        {
            "category": ["a"] * 8 + ["b"] * 8,
            "signal": [0, 1] * 8,
            "noise": np.linspace(0, 1, 16),
        }
    )
    train_y = np.array([0, 1] * 8, dtype=int)
    valid_x = pd.DataFrame(
        {
            "category": ["b", "a", "b", "a"],
            "signal": [1, 0, 0, 1],
            "noise": [0.2, 0.3, 0.4, 0.5],
        }
    )

    score = fit_predict_category_specialists(
        train_x,
        train_y,
        valid_x,
        seed=2026,
        max_iter=60,
        min_samples_leaf=2,
    )

    assert score.shape == (4,)
    assert np.isfinite(score).all()
    assert ((score >= 0) & (score <= 1)).all()
    assert score[0] > score[2]
    assert score[3] > score[1]


def test_category_specialists_fail_when_validation_category_has_no_train_model():
    train_x = pd.DataFrame({"category": ["a", "a", "a", "a"], "signal": [0, 1, 0, 1]})
    train_y = np.array([0, 1, 0, 1])
    valid_x = pd.DataFrame({"category": ["b"], "signal": [1]})

    try:
        fit_predict_category_specialists(train_x, train_y, valid_x, min_samples_leaf=1)
    except ValueError as exc:
        assert "category" in str(exc).lower()
    else:
        raise AssertionError("missing category specialist must fail")


def test_category_specialists_accept_source_weights_and_validate_length():
    train_x = pd.DataFrame(
        {
            "category": ["a"] * 12,
            "signal": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    # Contradictory low-weight pseudo labels say signal=0 is positive; the
    # high-weight authoritative rows say signal=1 is positive.
    train_y = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    weights = np.array([0.01, 0.01, 10, 10, 10, 10, 10, 10, 1, 1, 1, 1], dtype=float)
    valid_x = pd.DataFrame({"category": ["a", "a"], "signal": [0, 1]})

    weighted = fit_predict_category_specialists(
        train_x,
        train_y,
        valid_x,
        sample_weight=weights,
        max_iter=80,
        min_samples_leaf=2,
    )
    assert weighted[1] > weighted[0]

    try:
        fit_predict_category_specialists(
            train_x,
            train_y,
            valid_x,
            sample_weight=np.ones(3),
            min_samples_leaf=2,
        )
    except ValueError as exc:
        assert "sample_weight" in str(exc)
    else:
        raise AssertionError("wrong sample_weight length must fail")
