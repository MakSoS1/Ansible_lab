import pandas as pd

from ecup_matching.ml.v5_weak_specialists import forbidden_weak_item_ids


def test_weak_specialist_forbids_gold_and_held_fold_items_only():
    matches = pd.DataFrame(
        {
            "id1": [1, 3, 5, 7, 9, 11],
            "id2": [2, 4, 6, 8, 10, 12],
            "target": [1, 0, 1, 0, 1, 0],
        }
    )
    manifest = {
        "gold_rows": [4, 5],
        "fold_rows": [[0], [1], [2], [3]],
    }

    forbidden = forbidden_weak_item_ids(matches, manifest, held_fold=2)
    assert forbidden == {5, 6, 9, 10, 11, 12}
    assert forbidden.isdisjoint({1, 2, 3, 4, 7, 8})


def test_weak_specialist_rejects_unknown_fold_and_manifest_overlap():
    matches = pd.DataFrame({"id1": [1, 3], "id2": [2, 4], "target": [1, 0]})
    manifest = {"gold_rows": [1], "fold_rows": [[0]]}
    try:
        forbidden_weak_item_ids(matches, manifest, held_fold=7)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown held fold must fail")
