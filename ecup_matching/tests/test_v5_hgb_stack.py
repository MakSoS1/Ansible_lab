import numpy as np

from ecup_matching.ml.v5_hgb_stack import crossfit_fixed_hgb_stack


def _toy_scores(n: int) -> dict[str, np.ndarray]:
    x = np.linspace(-2.0, 2.0, n)
    return {
        "weak": x + 0.05 * np.sin(np.arange(n)),
        "sparse": 1.1 * x + 0.09 * np.cos(np.arange(n) * 0.2),
        "explicit": 0.8 * x + 0.2 * np.sin(np.arange(n) * 0.5),
        "contrastive": 0.5 * x + 0.4 * np.cos(np.arange(n) * 0.3),
        "teacher": 0.6 * x + 0.3 * np.sin(np.arange(n) * 0.4),
        "typed_explicit": x + 0.12 * np.cos(np.arange(n) * 0.8),
    }


def test_fixed_hgb_outer_prediction_does_not_use_outer_labels():
    n = 300
    scores = _toy_scores(n)
    folds = np.arange(n, dtype=np.int16) % 5
    categories = np.asarray(["a", "b", "c"] * (n // 3))
    interaction = np.where(categories == "a", scores["explicit"], scores["teacher"])
    target = (scores["sparse"] + 0.25 * interaction > 0.0).astype(np.int8)

    first = crossfit_fixed_hgb_stack(
        scores,
        target,
        categories,
        folds,
        max_iter=20,
        min_samples_leaf=10,
    )
    changed = target.copy()
    changed[folds == 0] = 1 - changed[folds == 0]
    second = crossfit_fixed_hgb_stack(
        scores,
        changed,
        categories,
        folds,
        max_iter=20,
        min_samples_leaf=10,
    )

    held = folds == 0
    assert np.allclose(first["oof_score"][held], second["oof_score"][held])
    assert np.isfinite(first["oof_score"]).all()
    assert set(first["fold_models"]) == {0, 1, 2, 3, 4}


def test_fixed_hgb_rejects_too_few_folds():
    n = 100
    scores = _toy_scores(n)
    target = (np.arange(n) % 2).astype(np.int8)
    categories = np.asarray(["a", "b"] * (n // 2))
    folds = np.zeros(n, dtype=np.int16)

    try:
        crossfit_fixed_hgb_stack(scores, target, categories, folds, max_iter=10)
    except ValueError as exc:
        assert "at least two folds" in str(exc)
    else:
        raise AssertionError("expected cross-fitting fold validation")
