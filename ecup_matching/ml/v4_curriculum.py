from __future__ import annotations

from typing import Iterable

import pandas as pd

from .weak_labels import prepare_weak_pairs, remove_human_conflicts, sample_weak_training


REQUIRED_COLUMNS = {
    "id1",
    "id2",
    "target",
    "category",
    "sample_weight",
    "text_a",
    "text_b",
    "source",
}


def _require_columns(frame: pd.DataFrame, label: str) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _item_ids(frame: pd.DataFrame) -> set[object]:
    return set(frame["id1"]) | set(frame["id2"])


def assert_item_disjoint(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    _require_columns(train, "train")
    _require_columns(validation, "validation")
    overlap = _item_ids(train) & _item_ids(validation)
    if overlap:
        raise RuntimeError(f"train/validation item overlap detected: {len(overlap)} items")


def build_human_curriculum(human: pd.DataFrame) -> pd.DataFrame:
    _require_columns(human, "human")
    source = human["source"].astype(str)
    if not source.eq("human").all():
        raise ValueError("full-human curriculum accepts authoritative human rows only")
    return human.copy().reset_index(drop=True)


def build_weak_curriculum(
    human: pd.DataFrame,
    weak: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    max_weak_rows: int,
    seed: int = 2026,
) -> pd.DataFrame:
    if max_weak_rows <= 0:
        raise ValueError("max_weak_rows must be positive")
    human_out = build_human_curriculum(human)
    _require_columns(weak, "weak")
    _require_columns(validation, "validation")

    prepared, _ = prepare_weak_pairs(weak)
    prepared, _ = remove_human_conflicts(
        prepared,
        human_out[["id1", "id2", "target"]],
    )
    validation_ids = _item_ids(validation)
    prepared = prepared.loc[
        ~prepared["id1"].isin(validation_ids) & ~prepared["id2"].isin(validation_ids)
    ].copy()
    if prepared.empty:
        raise RuntimeError("no eligible weak rows remain after leakage/conflict filtering")

    prepared["source"] = "weak"
    if "sample_weight" not in prepared.columns:
        prepared["sample_weight"] = prepared["weak_weight"].astype(float)
    else:
        prepared["sample_weight"] = prepared["weak_weight"].astype(float)
    selected = sample_weak_training(
        prepared,
        max_rows=min(max_weak_rows, len(prepared)),
        seed=seed,
        category_column="category",
    )
    result = pd.concat([human_out, selected], ignore_index=True)
    assert_item_disjoint(result, validation)
    return result.reset_index(drop=True)


def _sample_exact(frame: pd.DataFrame, n: int, seed: int, label: str) -> pd.DataFrame:
    if n < 0:
        raise ValueError("sample size must be non-negative")
    if n == 0:
        return frame.iloc[:0].copy()
    if len(frame) < n:
        raise ValueError(f"{label} needs at least {n} rows, got {len(frame)}")
    if len(frame) == n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed, replace=False)


def build_hard_replay_curriculum(
    parent: pd.DataFrame,
    mined_negatives: pd.DataFrame,
    positives: pd.DataFrame,
    *,
    total_rows: int,
    seed: int = 2026,
) -> pd.DataFrame:
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    for label, frame in (
        ("parent", parent),
        ("mined_negatives", mined_negatives),
        ("positives", positives),
    ):
        _require_columns(frame, label)

    if not (mined_negatives["target"].astype(float) < 0.5).all():
        raise ValueError("mined_negatives must contain negative targets only")
    if not (positives["target"].astype(float) >= 0.5).all():
        raise ValueError("positives must contain positive targets only")

    hard_n = total_rows // 4
    positive_n = total_rows // 4
    replay_n = total_rows - hard_n - positive_n

    hard = _sample_exact(mined_negatives, hard_n, seed + 1, "hard negatives")
    positive = _sample_exact(positives, positive_n, seed + 2, "positives")
    replay = _sample_exact(parent, replay_n, seed + 3, "ordinary replay")

    hard = hard.copy()
    positive = positive.copy()
    replay = replay.copy()
    hard["curriculum_role"] = "hard_negative"
    positive["curriculum_role"] = "positive"
    replay["curriculum_role"] = "replay"

    result = pd.concat([hard, positive, replay], ignore_index=True)
    if len(result) != total_rows:
        raise RuntimeError(f"expected {total_rows} replay rows, got {len(result)}")
    return result.sample(frac=1.0, random_state=seed + 4).reset_index(drop=True)
