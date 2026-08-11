import numpy as np

from ecup_matching.ml.v5_category_shrunk import crossfit_category_shrunk_simplex


def _toy_scores(n: int) -> dict[str, np.ndarray]:
    x = np.linspace(-2.0, 2.0, n)
    return {
        "weak": x + 0.08 * np.sin(np.arange(n)),
        "sparse": 1.15 * x + 0.09 * np.cos(np.arange(n)),
        "explicit": 0.85 * x + 0.13 * np.sin(np.arange(n) * 0.7),
        "contrastive": 0.5 * x + 0.35 * np.cos(np.arange(n) * 0.3),
        "teacher": 0.6 * x + 0.25 * np.sin(np.arange(n) * 0.5),
        "typed_explicit": x + 0.11 * np.cos(np.arange(n) * 0.9),
    }


def test_category_shrunk_outer_prediction_does_not_use_outer_labels():
    n = 120
    scores = _toy_scores(n)
    folds = np.arange(n, dtype=np.int16) % 5
    categories = np.asarray(["a", "b", "c"] * (n // 3))
    target = ((np.arange(n) // 3) % 2).astype(np.int8)

    first = crossfit_category_shrunk_simplex(
        scores,
        target,
        categories,
        folds,
        prior_strength=20.0,
        step_schedule=(0.2, 0.1),
        max_passes=2,
    )
    changed = target.copy()
    changed[folds == 0] = 1 - changed[folds == 0]
    second = crossfit_category_shrunk_simplex(
        scores,
        changed,
        categories,
        folds,
        prior_strength=20.0,
        step_schedule=(0.2, 0.1),
        max_passes=2,
    )

    held = folds == 0
    assert np.allclose(first["oof_score"][held], second["oof_score"][held])
    assert np.allclose(first["global_weights"][0], second["global_weights"][0])
    for category in ("a", "b", "c"):
        assert np.allclose(
            first["category_weights"][0][category],
            second["category_weights"][0][category],
        )


def test_category_shrunk_scores_every_row_and_keeps_simplex_weights():
    n = 120
    scores = _toy_scores(n)
    folds = np.arange(n, dtype=np.int16) % 5
    categories = np.asarray(["a", "b", "c"] * (n // 3))
    target = ((np.arange(n) // 4) % 2).astype(np.int8)

    result = crossfit_category_shrunk_simplex(
        scores,
        target,
        categories,
        folds,
        prior_strength=20.0,
        step_schedule=(0.2, 0.1),
        max_passes=2,
    )

    assert result["oof_score"].shape == (n,)
    assert np.isfinite(result["oof_score"]).all()
    assert set(result["global_weights"]) == {0, 1, 2, 3, 4}
    for fold, weights in result["global_weights"].items():
        assert np.all(weights >= 0.0)
        assert np.isclose(weights.sum(), 1.0)
        for category_weights in result["category_weights"][fold].values():
            assert np.all(category_weights >= 0.0)
            assert np.isclose(category_weights.sum(), 1.0)
