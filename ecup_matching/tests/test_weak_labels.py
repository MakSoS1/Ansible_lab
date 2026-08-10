import pandas as pd

from ecup_matching.ml.weak_labels import (
    prepare_weak_pairs,
    remove_human_conflicts,
    weak_confidence_weight,
)


def test_confidence_weight_exact_buckets():
    assert weak_confidence_weight(0.0) == 1.0
    assert weak_confidence_weight(0.03) == 1.0
    assert weak_confidence_weight(0.10) == 0.6
    assert weak_confidence_weight(0.15) == 0.6
    assert weak_confidence_weight(0.20) == 0.3
    assert weak_confidence_weight(0.30) == 0.3
    assert weak_confidence_weight(0.50) == 0.0
    assert weak_confidence_weight(0.70) == 0.3
    assert weak_confidence_weight(0.85) == 0.6
    assert weak_confidence_weight(0.97) == 1.0
    assert weak_confidence_weight(1.0) == 1.0


def test_prepare_weak_pairs_preserves_soft_target_and_excludes_mid_confidence():
    df = pd.DataFrame({"id1": [1, 2, 3], "id2": [4, 5, 6], "target": [0.01, 0.5, 0.8]})
    out, report = prepare_weak_pairs(df)
    assert out["target"].tolist() == [0.01, 0.8]
    assert out["weak_weight"].tolist() == [1.0, 0.3]
    assert out["hard_target"].tolist() == [0, 1]
    assert report["excluded_mid_confidence"] == 1


def test_remove_human_conflicts_drops_exact_human_pair_and_false_negative_inside_positive_component():
    human = pd.DataFrame(
        {
            "id1": [1, 2, 10],
            "id2": [2, 3, 11],
            "target": [1, 1, 0],
        }
    )
    weak = pd.DataFrame(
        {
            "id1": [2, 1, 1, 10, 20],
            "id2": [1, 3, 3, 11, 21],
            "target": [0.99, 0.01, 0.99, 0.99, 0.01],
            "weak_weight": [1.0] * 5,
            "hard_target": [1, 0, 1, 1, 0],
        }
    )
    out, report = remove_human_conflicts(weak, human)
    kept = set(out[["id1", "id2"]].itertuples(index=False, name=None))
    assert kept == {(1, 3), (20, 21)}
    assert report["exact_human_pairs_removed"] == 2
    assert report["component_false_negatives_removed"] == 1


def test_prepare_weak_pairs_is_deterministic():
    df = pd.DataFrame({"id1": [4, 1, 3], "id2": [2, 5, 6], "target": [0.98, 0.02, 0.75]})
    a, ra = prepare_weak_pairs(df)
    b, rb = prepare_weak_pairs(df)
    assert a.equals(b)
    assert ra == rb
