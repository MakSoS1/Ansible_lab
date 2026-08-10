import pandas as pd

from ecup_matching.ml.label_graph import (
    augment_transitive_positives,
    canonicalize_pairs,
    clean_human_pairs,
    positive_components,
)


def _pairs(rows):
    return pd.DataFrame(rows, columns=["id1", "id2", "target"])


def test_canonicalize_pairs_collapses_direction_deterministically():
    df = _pairs([(2, 1, 1), (1, 2, 1), (4, 3, 0)])
    out = canonicalize_pairs(df)
    assert list(out[["id1", "id2"]].itertuples(index=False, name=None)) == [(1, 2), (1, 2), (3, 4)]


def test_clean_human_pairs_drops_conflicting_pair_and_collapses_duplicates():
    df = _pairs([(1, 2, 1), (2, 1, 1), (3, 4, 0), (4, 3, 1), (5, 6, 0)])
    out, report = clean_human_pairs(df)
    assert set(out[["id1", "id2", "target"]].itertuples(index=False, name=None)) == {(1, 2, 1), (5, 6, 0)}
    assert report["duplicate_rows_removed"] == 1
    assert report["conflicting_pairs_dropped"] == 1


def test_positive_components_join_transitively():
    df = _pairs([(1, 2, 1), (2, 3, 1), (10, 11, 0)])
    comp = positive_components(df)
    assert comp[1] == comp[2] == comp[3]
    assert 10 not in comp and 11 not in comp


def test_transitive_positive_closure_adds_missing_edge_but_respects_explicit_negative():
    df = _pairs([(1, 2, 1), (2, 3, 1), (1, 3, 0), (7, 8, 1), (8, 9, 1)])
    out, report = augment_transitive_positives(df, max_pairs_per_component=100)
    triples = set(out[["id1", "id2", "target"]].itertuples(index=False, name=None))
    assert (1, 3, 1) not in triples
    assert (1, 3, 0) in triples
    assert (7, 9, 1) in triples
    assert report["transitive_positive_rows_added"] == 1


def test_transitive_closure_cap_is_deterministic():
    rows = [(i, i + 1, 1) for i in range(1, 8)]
    df = _pairs(rows)
    out1, report1 = augment_transitive_positives(df, max_pairs_per_component=3)
    out2, report2 = augment_transitive_positives(df, max_pairs_per_component=3)
    assert out1.equals(out2)
    assert report1 == report2
    assert report1["transitive_positive_rows_added"] == 3
