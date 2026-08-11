import pandas as pd

from ecup_matching.ml.run_v5_validation_audit import (
    build_split_descriptors,
    development_fold_indices,
)


def test_split_descriptors_bucket_label_free_difficulty_signals():
    pairs = pd.DataFrame(
        {
            "id1": [1, 2, 3],
            "id2": [11, 12, 13],
            "target": [1, 0, 1],
            "category": ["a", "a", "b"],
        }
    )
    features = pd.DataFrame(
        {
            "name_token_jaccard": [0.05, 0.55, 0.95],
            "name_char3_jaccard": [0.1, 0.6, 0.9],
            "model_code_conflict": [0.0, 1.0, 0.0],
            "number_conflict": [0.0, 1.0, 0.0],
            "quantity_conflict": [0.0, 0.0, 1.0],
            "attr_missing_any": [1.0, 0.0, 0.0],
            "hard_negative_score": [0.0, 0.7, 0.2],
        }
    )

    out = build_split_descriptors(pairs, features)
    assert out.columns.tolist() == [
        "category",
        "target",
        "lexical_bin",
        "char_bin",
        "model_conflict",
        "number_conflict",
        "quantity_conflict",
        "attr_missing",
        "hard_negative_bin",
    ]
    assert out["lexical_bin"].astype(str).nunique() == 3
    assert out["category"].tolist() == ["a", "a", "b"]
    assert out["target"].tolist() == [1, 0, 1]


def test_development_fold_indices_never_include_gold():
    manifest = {
        "gold_rows": [8, 9],
        "fold_rows": [[0, 1], [2, 3], [4, 5, 6, 7]],
    }
    train, valid = development_fold_indices(manifest, 1)
    assert valid.tolist() == [2, 3]
    assert train.tolist() == [0, 1, 4, 5, 6, 7]
    assert set(train).isdisjoint({8, 9})
    assert set(valid).isdisjoint({8, 9})
