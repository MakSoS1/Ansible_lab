import pandas as pd

from ecup_matching.ml.v5_sparse_crossfit import fit_transform_sparse_outer_fold


def test_sparse_outer_fold_vocabulary_is_fit_only_on_train_items():
    items = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["alpha common", "alpha common", "heldonlytoken common", "beta common"],
            "attributes": ["{}"] * 4,
            "category": ["a"] * 4,
        }
    )
    train_pairs = pd.DataFrame({"id1": [1], "id2": [2]})
    valid_pairs = pd.DataFrame({"id1": [3], "id2": [4]})

    result = fit_transform_sparse_outer_fold(
        items,
        train_pairs,
        valid_pairs,
        max_char_features=2000,
        max_word_features=1000,
    )

    assert result["train_features"].shape == (1, 4)
    assert result["valid_features"].shape == (1, 4)
    assert "heldonlytoken" not in result["name_word_vocabulary"]
    assert result["train_item_count"] == 2
    assert result["valid_item_count"] == 2


def test_sparse_outer_fold_supports_train_valid_item_overlap_without_duplicate_rows():
    items = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["alpha one", "alpha two", "alpha three"],
            "attributes": ["{}"] * 3,
            "category": ["a"] * 3,
        }
    )
    train_pairs = pd.DataFrame({"id1": [1], "id2": [2]})
    valid_pairs = pd.DataFrame({"id1": [2], "id2": [3]})
    result = fit_transform_sparse_outer_fold(items, train_pairs, valid_pairs)
    assert len(result["valid_features"]) == 1
