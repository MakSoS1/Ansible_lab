import numpy as np
import pandas as pd
import pandas.testing as pdt

from ecup_matching.ml.v8_graph import graph_features, graph_rescore


def _pairs():
    return pd.DataFrame(
        {
            "id1": ["a", "a", "b", "x", "x"],
            "id2": ["b", "c", "c", "y", "z"],
            "category": ["A", "A", "A", "B", "B"],
        }
    )


def test_reciprocal_best_and_degrees_use_only_observed_edges():
    frame = _pairs()
    scores = np.array([0.90, 0.80, 0.70, 0.60, 0.59])
    f = graph_features(frame, scores)
    # a-b is best incident edge for both a and b.
    assert bool(f.loc[0, "reciprocal_best"])
    assert f.loc[0, "degree_left"] == 2
    assert f.loc[0, "degree_right"] == 2
    assert f.loc[0, "rank_left"] == 1
    assert f.loc[0, "rank_right"] == 1
    # a-c is second for a, but c's best is a-c; it is not reciprocal-best.
    assert f.loc[1, "rank_left"] == 2
    assert f.loc[1, "rank_right"] == 1
    assert not bool(f.loc[1, "reciprocal_best"])


def test_category_isolation_prevents_same_item_id_from_cross_category_mixing():
    frame = pd.DataFrame(
        {"id1": [1, 1], "id2": [2, 3], "category": ["A", "B"]}
    )
    f = graph_features(frame, np.array([0.2, 0.9]))
    assert f["degree_left"].tolist() == [1, 1]
    assert f["rank_left"].tolist() == [1, 1]
    assert f["reciprocal_best"].tolist() == [True, True]


def test_graph_features_are_target_independent_and_permutation_equivariant():
    frame = _pairs()
    frame["target"] = [1, 0, 0, 1, 0]
    scores = np.array([0.90, 0.80, 0.70, 0.60, 0.59])
    base = graph_features(frame, scores)
    changed = frame.copy(); changed["target"] = 1 - changed["target"]
    pdt.assert_frame_equal(base, graph_features(changed, scores))

    order = np.array([3, 0, 4, 2, 1])
    perm = graph_features(frame.iloc[order].reset_index(drop=True), scores[order])
    restored = perm.iloc[np.argsort(order)].reset_index(drop=True)
    pdt.assert_frame_equal(base.reset_index(drop=True), restored, check_dtype=False)


def test_swapping_pair_endpoints_only_swaps_side_specific_features():
    frame = _pairs()
    scores = np.array([0.90, 0.80, 0.70, 0.60, 0.59])
    base = graph_features(frame, scores)
    swapped = frame.rename(columns={"id1": "id2", "id2": "id1"})[["id1", "id2", "category"]]
    other = graph_features(swapped, scores)
    assert np.array_equal(base["degree_left"], other["degree_right"])
    assert np.array_equal(base["degree_right"], other["degree_left"])
    assert np.array_equal(base["rank_left"], other["rank_right"])
    assert np.array_equal(base["rank_right"], other["rank_left"])
    assert np.array_equal(base["reciprocal_best"], other["reciprocal_best"])


def test_graph_rescore_changes_order_only_from_graph_context_and_stays_finite():
    # Equal base scores: reciprocal-best should outrank an ambiguous non-reciprocal edge.
    frame = pd.DataFrame(
        {
            "id1": ["a", "a", "b", "p", "p", "p"],
            "id2": ["b", "c", "c", "q", "r", "s"],
            "category": ["A"] * 6,
        }
    )
    scores = np.array([0.8, 0.8, 0.7, 0.8, 0.79, 0.78])
    features = graph_features(frame, scores)
    out = graph_rescore(scores, features, reciprocal_best_bonus=0.05, reciprocal_top3_bonus=0.01, ambiguity_penalty=0.01)
    assert out.shape == scores.shape
    assert np.isfinite(out).all()
    assert out[0] > out[3]


def test_graph_functions_do_not_require_target_column():
    frame = _pairs()
    scores = np.linspace(0.1, 0.9, len(frame))
    f = graph_features(frame, scores)
    out = graph_rescore(scores, f)
    assert len(out) == len(frame)
