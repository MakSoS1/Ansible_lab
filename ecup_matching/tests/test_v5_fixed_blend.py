import numpy as np
import pytest

from ecup_matching.ml.v5_fixed_blend import fixed_blend_candidates, percentile_rank


def test_percentile_rank_is_finite_monotonic_bounded_and_tie_stable():
    values = np.array([10.0, 5.0, 5.0, 20.0])
    ranks = percentile_rank(values)

    assert ranks.shape == values.shape
    assert np.isfinite(ranks).all()
    assert np.all((0.0 <= ranks) & (ranks <= 1.0))
    assert ranks[1] == pytest.approx(ranks[2])
    assert ranks[3] > ranks[0] > ranks[1]


def test_fixed_blend_candidates_are_order_independent_and_target_free():
    branch_scores = {
        "category": np.array([0.1, 0.7, 0.4, 0.9]),
        "weak": np.array([0.2, 0.8, 0.3, 0.7]),
        "sparse": np.array([0.05, 0.95, 0.5, 0.6]),
        "explicit": np.array([0.15, 0.85, 0.45, 0.8]),
    }
    cosine = np.array([-0.2, 0.8, 0.1, 0.5])

    first = fixed_blend_candidates(branch_scores, contrastive_cosine=cosine)
    reversed_sources = dict(reversed(list(branch_scores.items())))
    second = fixed_blend_candidates(reversed_sources, contrastive_cosine=cosine)

    assert set(first) == {"prob_mean_4", "rank_mean_3", "rank_mean_4", "rank_mean_5"}
    for name in first:
        assert first[name].shape == (4,)
        assert np.isfinite(first[name]).all()
        assert np.allclose(first[name], second[name])


def test_fixed_blend_identical_rankings_preserve_ranking():
    common = np.array([0.05, 0.3, 0.2, 0.95, 0.7])
    sources = {name: common.copy() for name in ("category", "weak", "sparse", "explicit")}
    result = fixed_blend_candidates(sources, contrastive_cosine=common)
    expected_order = np.argsort(common)

    for score in result.values():
        assert np.array_equal(np.argsort(score), expected_order)


def test_fixed_blend_rejects_missing_nonfinite_or_misaligned_inputs():
    good = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="missing required"):
        fixed_blend_candidates({"category": good})

    bad = {name: good.copy() for name in ("category", "weak", "sparse", "explicit")}
    bad["weak"] = np.array([0.1, np.nan])
    with pytest.raises(ValueError, match="finite"):
        fixed_blend_candidates(bad)

    bad = {name: good.copy() for name in ("category", "weak", "sparse", "explicit")}
    bad["sparse"] = np.array([0.1])
    with pytest.raises(ValueError, match="equal length"):
        fixed_blend_candidates(bad)
