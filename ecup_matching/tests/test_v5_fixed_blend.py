import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.run_v5_fixed_blend import align_oof_frame
from ecup_matching.ml.v5_fixed_blend import (
    fixed_blend_candidates,
    grouped_percentile_rank,
    leave_one_out_rank_candidates,
    orthogonal_rank_candidates,
    percentile_rank,
    rank_ablation_candidates,
    typed_explicit_rank_candidates,
)


def test_percentile_rank_is_finite_monotonic_bounded_and_tie_stable():
    values = np.array([10.0, 5.0, 5.0, 20.0])
    ranks = percentile_rank(values)

    assert ranks.shape == values.shape
    assert np.isfinite(ranks).all()
    assert np.all((0.0 <= ranks) & (ranks <= 1.0))
    assert ranks[1] == pytest.approx(ranks[2])
    assert ranks[3] > ranks[0] > ranks[1]


def test_grouped_percentile_rank_ignores_cross_category_scale():
    values = np.array([1.0, 2.0, 100.0, 200.0])
    groups = np.array(["a", "a", "b", "b"])
    ranks = grouped_percentile_rank(values, groups)

    assert ranks.tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])

    rescaled = values.copy()
    rescaled[groups == "b"] *= 1000.0
    assert grouped_percentile_rank(rescaled, groups).tolist() == pytest.approx(ranks.tolist())


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


def test_rank_ablation_candidates_are_predeclared_target_free_and_group_aligned():
    scores = {
        "category": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        "weak": np.array([0.2, 0.8, 0.4, 0.6, 0.1, 0.9]),
        "sparse": np.array([0.3, 0.7, 0.5, 0.4, 0.2, 0.95]),
        "explicit": np.array([0.25, 0.75, 0.45, 0.55, 0.15, 0.85]),
    }
    cosine = np.array([-0.2, 0.5, 0.1, 0.4, -0.1, 0.8])
    groups = np.array(["a", "a", "b", "b", "c", "c"])

    result = rank_ablation_candidates(scores, groups=groups, contrastive_cosine=cosine)
    assert set(result) == {
        "global_rank_mean_4_no_category",
        "global_rank_mean_3_strong",
        "category_rank_mean_5",
        "category_rank_mean_4_no_category",
        "category_rank_mean_3_strong",
    }
    for values in result.values():
        assert values.shape == (6,)
        assert np.isfinite(values).all()
        assert np.all((0.0 <= values) & (values <= 1.0))


def test_orthogonal_rank_candidates_match_predeclared_source_sets_and_are_target_free():
    current4 = {
        "weak": np.array([0.2, 0.8, 0.4, 0.6]),
        "sparse": np.array([0.3, 0.7, 0.5, 0.4]),
        "explicit": np.array([0.25, 0.75, 0.45, 0.55]),
        "contrastive": np.array([-0.2, 0.5, 0.1, 0.4]),
    }
    extras = {
        "teacher2_raw": np.array([-1.0, 2.0, 0.0, 0.5]),
        "weighted": np.array([0.1, 0.9, 0.2, 0.7]),
        "pretrained_raw": np.array([-0.1, 0.4, 0.3, 0.2]),
    }

    result = orthogonal_rank_candidates(current4, extras)
    assert set(result) == {
        "current4_plus_teacher",
        "current4_plus_weighted",
        "current4_plus_pretrained",
        "current4_plus_teacher_weighted",
        "current4_plus_all_three",
    }
    for values in result.values():
        assert values.shape == (4,)
        assert np.isfinite(values).all()
        assert np.all((0.0 <= values) & (values <= 1.0))

    reversed_result = orthogonal_rank_candidates(
        dict(reversed(list(current4.items()))),
        dict(reversed(list(extras.items()))),
    )
    for name in result:
        assert np.allclose(result[name], reversed_result[name])


def test_leave_one_out_rank_candidates_are_exactly_the_five_predeclared_removals():
    five = {
        "weak": np.array([0.2, 0.8, 0.4, 0.6]),
        "sparse": np.array([0.3, 0.7, 0.5, 0.4]),
        "explicit": np.array([0.25, 0.75, 0.45, 0.55]),
        "contrastive": np.array([-0.2, 0.5, 0.1, 0.4]),
        "teacher": np.array([-1.0, 2.0, 0.0, 0.5]),
    }
    result = leave_one_out_rank_candidates(five)
    assert set(result) == {
        "loo_drop_weak",
        "loo_drop_sparse",
        "loo_drop_explicit",
        "loo_drop_contrastive",
        "loo_drop_teacher",
    }
    for values in result.values():
        assert values.shape == (4,)
        assert np.isfinite(values).all()
        assert np.all((0.0 <= values) & (values <= 1.0))

    expected_drop_teacher = np.mean(
        np.vstack([percentile_rank(five[name]) for name in ("weak", "sparse", "explicit", "contrastive")]),
        axis=0,
    )
    assert np.allclose(result["loo_drop_teacher"], expected_drop_teacher)


def test_typed_explicit_rank_candidates_are_exactly_add_and_replace():
    current5 = {
        "weak": np.array([0.2, 0.8, 0.4, 0.6]),
        "sparse": np.array([0.3, 0.7, 0.5, 0.4]),
        "explicit": np.array([0.25, 0.75, 0.45, 0.55]),
        "contrastive": np.array([-0.2, 0.5, 0.1, 0.4]),
        "teacher": np.array([-1.0, 2.0, 0.0, 0.5]),
    }
    typed = np.array([0.15, 0.85, 0.35, 0.65])
    result = typed_explicit_rank_candidates(current5, typed)
    assert set(result) == {
        "current5_plus_typed_explicit",
        "current5_replace_explicit_with_typed",
    }
    for values in result.values():
        assert values.shape == (4,)
        assert np.isfinite(values).all()
        assert np.all((0.0 <= values) & (values <= 1.0))

    expected_replace = np.mean(
        np.vstack(
            [percentile_rank(current5[name]) for name in ("weak", "sparse", "contrastive", "teacher")]
            + [percentile_rank(typed)]
        ),
        axis=0,
    )
    assert np.allclose(result["current5_replace_explicit_with_typed"], expected_replace)


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

    with pytest.raises(ValueError, match="equal length"):
        grouped_percentile_rank(np.array([0.1, 0.2]), np.array(["a"]))

    with pytest.raises(ValueError, match="missing required"):
        orthogonal_rank_candidates(
            {"weak": good, "sparse": good, "explicit": good, "contrastive": good},
            {"teacher2_raw": good},
        )

    with pytest.raises(ValueError, match="missing required"):
        leave_one_out_rank_candidates({"weak": good})

    with pytest.raises(ValueError, match="missing required"):
        typed_explicit_rank_candidates({"weak": good}, good)


def test_align_oof_frame_requires_exact_rows_and_folds(tmp_path):
    expected_rows = np.array([10, 20, 30], dtype=np.int64)
    expected_folds = np.array([0, 1, 0], dtype=np.int16)
    good = pd.DataFrame(
        {"row_index": [30, 10, 20], "fold": [0, 0, 1], "score": [0.3, 0.1, 0.2]}
    )
    good_path = tmp_path / "good.parquet"
    good.to_parquet(good_path, index=False)

    aligned = align_oof_frame(
        [good_path],
        expected_rows=expected_rows,
        expected_folds=expected_folds,
        required_columns=("score",),
        source_name="toy",
    )
    assert aligned["row_index"].tolist() == [10, 20, 30]
    assert aligned["fold"].tolist() == [0, 1, 0]

    duplicate = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    duplicate_path = tmp_path / "duplicate.parquet"
    duplicate.to_parquet(duplicate_path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        align_oof_frame(
            [duplicate_path],
            expected_rows=expected_rows,
            expected_folds=expected_folds,
            required_columns=("score",),
            source_name="toy",
        )

    wrong_fold = good.copy()
    wrong_fold.loc[wrong_fold["row_index"] == 20, "fold"] = 0
    wrong_fold_path = tmp_path / "wrong-fold.parquet"
    wrong_fold.to_parquet(wrong_fold_path, index=False)
    with pytest.raises(ValueError, match="fold"):
        align_oof_frame(
            [wrong_fold_path],
            expected_rows=expected_rows,
            expected_folds=expected_folds,
            required_columns=("score",),
            source_name="toy",
        )

    missing = good[good["row_index"] != 20]
    missing_path = tmp_path / "missing.parquet"
    missing.to_parquet(missing_path, index=False)
    with pytest.raises(ValueError, match="cover"):
        align_oof_frame(
            [missing_path],
            expected_rows=expected_rows,
            expected_folds=expected_folds,
            required_columns=("score",),
            source_name="toy",
        )
