import pandas as pd
import pytest

from ecup_matching.ml.v3_dataset import assert_no_item_overlap, sample_v3_training_pairs


def _human_frame() -> pd.DataFrame:
    rows = [
        {"id1": 1, "id2": 2, "target": 1.0, "category": "Электроника"},
        {"id1": 3, "id2": 4, "target": 1.0, "category": "Одежда"},
    ]
    for idx in range(10):
        rows.append(
            {
                "id1": 100 + idx * 2,
                "id2": 101 + idx * 2,
                "target": 0.0,
                "category": "Электроника" if idx < 5 else "Аптека",
            }
        )
    return pd.DataFrame(rows)


def _weak_frame() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        rows.append(
            {
                "id1": 300 + idx * 2,
                "id2": 301 + idx * 2,
                "target": 0.99 if idx % 2 else 0.01,
                "category": "Электроника" if idx < 6 else "Аптека",
                "weak_weight": 1.0,
            }
        )
    # Must be removed because one item belongs to fixed validation.
    rows.append(
        {
            "id1": 900,
            "id2": 999,
            "target": 0.99,
            "category": "Электроника",
            "weak_weight": 1.0,
        }
    )
    return pd.DataFrame(rows)


def test_sample_v3_training_pairs_is_deterministic_and_preserves_human_positives():
    kwargs = dict(
        human_pairs=_human_frame(),
        weak_pairs=_weak_frame(),
        validation_item_ids={900, 901},
        max_rows=10,
        priority_categories={"Электроника"},
        priority_fraction=0.60,
        seed=2026,
    )

    first = sample_v3_training_pairs(**kwargs)
    second = sample_v3_training_pairs(**kwargs)

    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert len(first) == 10
    positives = first[(first["source"] == "human") & (first["target"] >= 0.5)]
    assert set(map(tuple, positives[["id1", "id2"]].to_numpy())) == {(1, 2), (3, 4)}


def test_sample_v3_training_pairs_excludes_every_validation_item():
    sampled = sample_v3_training_pairs(
        human_pairs=_human_frame(),
        weak_pairs=_weak_frame(),
        validation_item_ids={900, 901},
        max_rows=12,
        priority_categories={"Электроника"},
        priority_fraction=0.60,
        seed=7,
    )

    assert 900 not in set(sampled["id1"]) | set(sampled["id2"])
    assert 901 not in set(sampled["id1"]) | set(sampled["id2"])


def test_priority_categories_receive_at_least_requested_extra_budget_when_available():
    sampled = sample_v3_training_pairs(
        human_pairs=_human_frame(),
        weak_pairs=_weak_frame(),
        validation_item_ids={900, 901},
        max_rows=10,
        priority_categories={"Электроника"},
        priority_fraction=0.60,
        seed=11,
    )

    mandatory_human_positives = sampled[(sampled["source"] == "human") & (sampled["target"] >= 0.5)]
    discretionary = sampled.drop(index=mandatory_human_positives.index)
    priority_count = int(discretionary["category"].isin({"Электроника"}).sum())
    assert priority_count >= 5  # ceil(8 discretionary rows * 0.60)


def test_assert_no_item_overlap_fails_closed():
    train = pd.DataFrame({"id1": [1, 5], "id2": [2, 6]})
    valid = pd.DataFrame({"id1": [3, 5], "id2": [4, 7]})

    with pytest.raises(RuntimeError, match="overlapping item IDs"):
        assert_no_item_overlap(train, valid)
