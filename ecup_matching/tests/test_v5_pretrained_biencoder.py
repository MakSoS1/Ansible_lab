import numpy as np

from ecup_matching.ml.run_v5_pretrained_biencoder import development_rows_and_folds


def test_development_rows_and_folds_exclude_gold_and_preserve_original_indices():
    manifest = {
        "gold_rows": [8, 9, 10],
        "fold_rows": [[0, 3], [1, 4], [2, 5], [6], [7]],
    }
    rows, folds = development_rows_and_folds(manifest, total_rows=11)

    assert rows.tolist() == list(range(8))
    assert folds.tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    assert set(rows).isdisjoint({8, 9, 10})


def test_development_rows_and_folds_reject_overlap_or_missing_coverage():
    bad_overlap = {"gold_rows": [3], "fold_rows": [[0, 1], [1, 2]]}
    try:
        development_rows_and_folds(bad_overlap, total_rows=4)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate development row must fail")

    bad_missing = {"gold_rows": [3], "fold_rows": [[0], [1]]}
    try:
        development_rows_and_folds(bad_missing, total_rows=4)
    except ValueError:
        pass
    else:
        raise AssertionError("manifest that does not cover all rows must fail")
