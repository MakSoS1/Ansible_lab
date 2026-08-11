from __future__ import annotations

import pandas as pd
import pytest

from ecup_matching.ml.v4_curriculum import (
    assert_item_disjoint,
    build_hard_replay_curriculum,
    build_human_curriculum,
    build_weak_curriculum,
)


def _frame(rows: list[tuple[int, int, float, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id1": id1,
                "id2": id2,
                "target": target,
                "category": category,
                "sample_weight": 1.0,
                "text_a": f"item {id1}",
                "text_b": f"item {id2}",
                "source": source,
            }
            for id1, id2, target, category, source in rows
        ]
    )


def test_full_human_curriculum_keeps_every_authoritative_row() -> None:
    human = _frame(
        [
            (1, 2, 1.0, "Электроника", "human"),
            (3, 4, 0.0, "Электроника", "human"),
            (5, 6, 1.0, "Одежда", "human"),
            (7, 8, 0.0, "Мебель", "human"),
        ]
    )

    result = build_human_curriculum(human)

    assert len(result) == len(human)
    assert set(result["source"]) == {"human"}
    assert set(zip(result["id1"], result["id2"])) == set(zip(human["id1"], human["id2"]))


def test_item_disjoint_guard_rejects_any_shared_item() -> None:
    train = _frame([(1, 2, 1.0, "Электроника", "human")])
    valid = _frame([(2, 3, 0.0, "Электроника", "human")])

    with pytest.raises(RuntimeError, match="overlap"):
        assert_item_disjoint(train, valid)


def test_weak_curriculum_is_disjoint_capped_and_deterministic() -> None:
    human = _frame(
        [
            (1, 2, 1.0, "Электроника", "human"),
            (3, 4, 0.0, "Одежда", "human"),
        ]
    )
    weak = _frame(
        [
            (10, 11, 0.99, "Электроника", "weak"),
            (12, 13, 0.01, "Электроника", "weak"),
            (14, 15, 0.90, "Одежда", "weak"),
            (16, 17, 0.10, "Одежда", "weak"),
            (18, 19, 0.80, "Мебель", "weak"),
            (20, 21, 0.20, "Мебель", "weak"),
            (22, 23, 0.99, "Обувь", "weak"),
            (24, 25, 0.01, "Обувь", "weak"),
            (30, 99, 0.99, "Электроника", "weak"),
        ]
    )
    valid = _frame([(99, 100, 1.0, "Электроника", "human")])

    first = build_weak_curriculum(human, weak, valid, max_weak_rows=6, seed=2026)
    second = build_weak_curriculum(human, weak, valid, max_weak_rows=6, seed=2026)

    assert len(first) == len(human) + 6
    assert int((first["source"] == "weak").sum()) == 6
    assert 99 not in set(first["id1"]) | set(first["id2"])
    assert first[["id1", "id2", "source"]].equals(second[["id1", "id2", "source"]])
    assert_item_disjoint(first, valid)


def test_hard_replay_curriculum_uses_25_25_50_mix() -> None:
    parent = _frame(
        [
            (100 + i * 2, 101 + i * 2, float(i % 2), "Электроника", "weak" if i >= 8 else "human")
            for i in range(16)
        ]
    )
    hard = _frame(
        [
            (1000 + i * 2, 1001 + i * 2, 0.0, "Электроника", "human")
            for i in range(8)
        ]
    )
    positives = _frame(
        [
            (2000 + i * 2, 2001 + i * 2, 1.0, "Электроника", "human")
            for i in range(8)
        ]
    )

    result = build_hard_replay_curriculum(
        parent,
        hard,
        positives,
        total_rows=16,
        seed=2026,
    )

    assert len(result) == 16
    assert int((result["curriculum_role"] == "hard_negative").sum()) == 4
    assert int((result["curriculum_role"] == "positive").sum()) == 4
    assert int((result["curriculum_role"] == "replay").sum()) == 8
    assert (result.loc[result["curriculum_role"] == "hard_negative", "target"] < 0.5).all()
    assert (result.loc[result["curriculum_role"] == "positive", "target"] >= 0.5).all()
