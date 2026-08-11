import numpy as np
import pandas as pd

from ecup_matching.ml.v5_contrastive_data import select_fold_contrastive_pairs


def test_contrastive_selector_uses_only_outer_train_rows_and_preserves_all_positives():
    rows = []
    folds = []
    base = []
    for fold in range(5):
        for i in range(20):
            target = int(i % 4 == 0)
            rows.append({"id1": fold * 1000 + i * 2, "id2": fold * 1000 + i * 2 + 1, "target": target, "category": "a"})
            folds.append(fold)
            # Make some negatives hard according to an OOF anchor.
            base.append(0.9 if (target == 0 and i % 3 == 0) else (0.8 if target else 0.1))
    frame = pd.DataFrame(rows)

    selected = select_fold_contrastive_pairs(
        frame,
        np.asarray(folds),
        np.asarray(base),
        held_fold=2,
        max_negative_to_positive=2.0,
        hard_negative_fraction=0.5,
        seed=2026,
    )

    assert not (selected["_source_row"].isin(np.flatnonzero(np.asarray(folds) == 2))).any()
    train_positive_rows = set(np.flatnonzero((np.asarray(folds) != 2) & (frame["target"].to_numpy() == 1)).tolist())
    selected_positive_rows = set(selected.loc[selected["target"] == 1, "_source_row"].tolist())
    assert selected_positive_rows == train_positive_rows
    negatives = int((selected["target"] == 0).sum())
    positives = int((selected["target"] == 1).sum())
    assert negatives <= 2 * positives
    assert selected["_source_row"].is_unique


def test_contrastive_selector_is_deterministic_and_rejects_invalid_fold():
    frame = pd.DataFrame(
        {
            "id1": np.arange(20),
            "id2": np.arange(100, 120),
            "target": [0, 1] * 10,
            "category": ["a"] * 20,
        }
    )
    folds = np.repeat(np.arange(5), 4)
    base = np.linspace(0.01, 0.99, 20)
    a = select_fold_contrastive_pairs(frame, folds, base, held_fold=1, seed=17)
    b = select_fold_contrastive_pairs(frame, folds, base, held_fold=1, seed=17)
    assert a.equals(b)

    try:
        select_fold_contrastive_pairs(frame, folds, base, held_fold=9)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown held fold must fail")
