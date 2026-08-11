import numpy as np
import pytest

from ecup_matching.ml.v5_meta_blend import (
    SIX_SIGNAL_NAMES,
    crossfit_global_simplex_blend,
    fit_simplex_weights,
)


def _toy_scores(n: int) -> dict[str, np.ndarray]:
    x = np.linspace(-2.0, 2.0, n)
    return {
        "weak": x + 0.05 * np.sin(np.arange(n)),
        "sparse": 1.2 * x + 0.08 * np.cos(np.arange(n)),
        "explicit": 0.9 * x + 0.12 * np.sin(np.arange(n) * 0.7),
        "contrastive": 0.45 * x + 0.4 * np.cos(np.arange(n) * 0.3),
        "teacher": 0.55 * x + 0.3 * np.sin(np.arange(n) * 0.5),
        "typed_explicit": 1.05 * x + 0.1 * np.cos(np.arange(n) * 0.9),
    }


def test_fit_simplex_weights_returns_deterministic_nonnegative_simplex():
    n = 48
    scores = _toy_scores(n)
    matrix = np.column_stack([scores[name] for name in SIX_SIGNAL_NAMES])
    target = np.tile(np.array([0, 0, 1, 1], dtype=np.int8), n // 4)
    categories = np.where(np.arange(n) % 2 == 0, "a", "b")

    first = fit_simplex_weights(
        matrix,
        target,
        categories,
        step_schedule=(0.2, 0.1, 0.05),
        max_passes=4,
    )
    second = fit_simplex_weights(
        matrix,
        target,
        categories,
        step_schedule=(0.2, 0.1, 0.05),
        max_passes=4,
    )

    assert first.shape == (6,)
    assert np.isfinite(first).all()
    assert np.all(first >= 0.0)
    assert first.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(first, second)


def test_crossfit_prediction_for_fold_does_not_use_that_folds_labels():
    n = 72
    scores = _toy_scores(n)
    folds = np.arange(n, dtype=np.int16) % 3
    categories = np.where(np.arange(n) % 2 == 0, "a", "b")
    target = ((np.arange(n) // 2) % 2).astype(np.int8)

    first = crossfit_global_simplex_blend(
        scores,
        target,
        categories,
        folds,
        step_schedule=(0.2, 0.1),
        max_passes=3,
    )

    changed = target.copy()
    changed[folds == 0] = 1 - changed[folds == 0]
    second = crossfit_global_simplex_blend(
        scores,
        changed,
        categories,
        folds,
        step_schedule=(0.2, 0.1),
        max_passes=3,
    )

    held_out = folds == 0
    assert np.allclose(first["oof_score"][held_out], second["oof_score"][held_out])
    assert np.allclose(first["fold_weights"][0], second["fold_weights"][0])


def test_crossfit_uses_all_rows_once_and_weights_are_simplex():
    n = 60
    scores = _toy_scores(n)
    folds = np.arange(n, dtype=np.int16) % 5
    categories = np.where(np.arange(n) % 3 == 0, "a", "b")
    target = ((np.arange(n) // 3) % 2).astype(np.int8)

    result = crossfit_global_simplex_blend(
        scores,
        target,
        categories,
        folds,
        step_schedule=(0.2, 0.1),
        max_passes=2,
    )

    assert result["oof_score"].shape == (n,)
    assert np.isfinite(result["oof_score"]).all()
    assert set(result["fold_weights"]) == {0, 1, 2, 3, 4}
    for weights in result["fold_weights"].values():
        assert np.all(weights >= 0.0)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_crossfit_rejects_missing_signal_or_nonfinite_score():
    n = 24
    scores = _toy_scores(n)
    target = (np.arange(n) % 2).astype(np.int8)
    categories = np.where(np.arange(n) % 2 == 0, "a", "b")
    folds = np.arange(n, dtype=np.int16) % 3

    missing = dict(scores)
    missing.pop("teacher")
    with pytest.raises(ValueError, match="missing required score sources"):
        crossfit_global_simplex_blend(missing, target, categories, folds)

    bad = dict(scores)
    bad["weak"] = bad["weak"].copy()
    bad["weak"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        crossfit_global_simplex_blend(bad, target, categories, folds)
