import numpy as np

from ecup_matching.ml.v5_fixed_blend import normalized_retrain_rank_candidates, percentile_rank


def test_normalized_retrain_candidates_are_exactly_the_four_frozen_variants():
    base5 = {
        "weak": np.array([0.1, 0.9, 0.3, 0.7]),
        "sparse": np.array([0.2, 0.8, 0.4, 0.6]),
        "explicit": np.array([0.25, 0.75, 0.35, 0.65]),
        "contrastive": np.array([-0.3, 0.7, 0.0, 0.4]),
        "teacher": np.array([-1.0, 2.0, 0.2, 0.5]),
    }
    typed_explicit = np.array([0.15, 0.85, 0.45, 0.55])
    normalized_explicit = np.array([0.12, 0.88, 0.5, 0.52])
    normalized_category = np.array([0.18, 0.82, 0.38, 0.62])

    result = normalized_retrain_rank_candidates(
        base5,
        typed_explicit_scores=typed_explicit,
        normalized_explicit_scores=normalized_explicit,
        normalized_category_scores=normalized_category,
    )

    assert set(result) == {
        "current6_plus_normalized_explicit",
        "current6_replace_typed_with_normalized_explicit",
        "current6_replace_old_explicit_with_normalized_explicit",
        "current6_plus_normalized_category",
    }
    for values in result.values():
        assert values.shape == (4,)
        assert np.isfinite(values).all()
        assert np.all((0.0 <= values) & (values <= 1.0))

    ranks = {name: percentile_rank(values) for name, values in base5.items()}
    typed_rank = percentile_rank(typed_explicit)
    new_rank = percentile_rank(normalized_explicit)
    expected_add = np.mean(np.vstack([*ranks.values(), typed_rank, new_rank]), axis=0)
    assert np.allclose(result["current6_plus_normalized_explicit"], expected_add)


def test_normalized_retrain_candidates_do_not_depend_on_mapping_order():
    base5 = {
        "weak": np.array([0.1, 0.9]),
        "sparse": np.array([0.2, 0.8]),
        "explicit": np.array([0.3, 0.7]),
        "contrastive": np.array([-0.2, 0.5]),
        "teacher": np.array([-1.0, 1.0]),
    }
    kwargs = dict(
        typed_explicit_scores=np.array([0.35, 0.65]),
        normalized_explicit_scores=np.array([0.4, 0.6]),
        normalized_category_scores=np.array([0.45, 0.55]),
    )
    first = normalized_retrain_rank_candidates(base5, **kwargs)
    second = normalized_retrain_rank_candidates(dict(reversed(list(base5.items()))), **kwargs)
    for name in first:
        assert np.allclose(first[name], second[name])
