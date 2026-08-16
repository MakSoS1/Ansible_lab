"""Two-hop corroboration: the graph property `v8_graph` never expressed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.v16_graph_transitive import (
    graph_degree_report,
    rescore_with_transitivity,
    transitive_features,
)


def _triangle() -> pd.DataFrame:
    # a-b, b-c, a-c inside one category.
    return pd.DataFrame(
        {
            "id1": ["a", "b", "a"],
            "id2": ["b", "c", "c"],
            "category": ["Электроника"] * 3,
        }
    )


def test_closed_triangle_corroborates_every_edge():
    frame = _triangle()
    out = transitive_features(frame, [0.9, 0.8, 0.2])
    # a-b is supported through c by min(s(a,c), s(b,c)) = min(0.2, 0.8) = 0.2
    assert out["two_hop_support"].tolist() == pytest.approx([0.2, 0.2, 0.8])
    assert out["shared_neighbours"].tolist() == [1, 1, 1]
    # The weak direct edge a-c has the strongest corroboration.
    assert out["support_minus_direct"].iloc[2] == pytest.approx(0.6)


def test_edge_without_shared_neighbour_has_zero_support_and_is_distinguishable():
    frame = pd.DataFrame(
        {"id1": ["a", "c"], "id2": ["b", "d"], "category": ["x", "x"]}
    )
    out = transitive_features(frame, [0.9, 0.9])
    assert out["two_hop_support"].tolist() == [0.0, 0.0]
    assert out["shared_neighbours"].tolist() == [0, 0]
    assert out["has_two_hop"].tolist() == [0.0, 0.0]


def test_paths_never_cross_a_category_boundary():
    """An identifier reused in another category must not create a path."""
    frame = pd.DataFrame(
        {
            "id1": ["a", "b", "a"],
            "id2": ["b", "c", "c"],
            "category": ["x", "y", "x"],
        }
    )
    out = transitive_features(frame, [0.9, 0.9, 0.9])
    # b-c lives in category y alone, so nothing in x can route through it.
    assert out["shared_neighbours"].tolist() == [0, 0, 0]


def test_support_uses_the_bottleneck_not_the_average():
    frame = pd.DataFrame(
        {"id1": ["a", "b", "a"], "id2": ["b", "c", "c"], "category": ["x"] * 3}
    )
    out = transitive_features(frame, [0.5, 1.0, 0.1])
    # Path a-c-b has bottleneck min(0.1, 1.0) = 0.1, not the mean 0.55.
    assert out["two_hop_support"].iloc[0] == pytest.approx(0.1)


def test_best_of_several_paths_wins():
    frame = pd.DataFrame(
        {
            "id1": ["a", "a", "b", "a", "b"],
            "id2": ["b", "c", "c", "d", "d"],
            "category": ["x"] * 5,
        }
    )
    out = transitive_features(frame, [0.5, 0.2, 0.3, 0.7, 0.8])
    # a-b is supported through c (min 0.2) and through d (min 0.7); best is 0.7.
    assert out["two_hop_support"].iloc[0] == pytest.approx(0.7)
    assert out["shared_neighbours"].iloc[0] == 2


def test_features_reject_misaligned_or_nonfinite_input():
    frame = _triangle()
    with pytest.raises(ValueError, match="aligned"):
        transitive_features(frame, [0.1, 0.2])
    with pytest.raises(ValueError, match="NaN"):
        transitive_features(frame, [0.1, np.nan, 0.3])
    with pytest.raises(ValueError, match="missing columns"):
        transitive_features(frame.drop(columns=["category"]), [0.1, 0.2, 0.3])


def test_degree_report_exposes_a_degenerate_graph():
    """The check that decides whether reciprocal-best can carry any signal."""
    degenerate = pd.DataFrame(
        {"id1": ["a", "c", "e"], "id2": ["b", "d", "f"], "category": ["x"] * 3}
    )
    report = graph_degree_report(degenerate)
    assert report["fraction_degree_1"] == 1.0
    assert report["degree_max"] == 1.0

    dense = pd.DataFrame(
        {
            "id1": ["a", "a", "a", "b"],
            "id2": ["b", "c", "d", "c"],
            "category": ["x"] * 4,
        }
    )
    dense_report = graph_degree_report(dense)
    assert dense_report["degree_max"] == 3.0
    assert dense_report["fraction_degree_1"] < 1.0
    assert "x" in dense_report["per_category"]


def test_rescore_is_a_noop_until_the_caller_opts_in():
    frame = _triangle()
    features = transitive_features(frame, [0.9, 0.8, 0.2])
    base = np.array([0.5, 0.4, 0.3])
    unchanged = rescore_with_transitivity(base, frame, features)
    assert np.array_equal(unchanged, base)

    boosted = rescore_with_transitivity(base, frame, features, support_weight=0.1)
    assert boosted[2] > base[2]

    penalised = rescore_with_transitivity(
        base, frame, features, orphan_penalty=0.05
    )
    assert np.array_equal(penalised, base), "every edge here has a shared neighbour"
