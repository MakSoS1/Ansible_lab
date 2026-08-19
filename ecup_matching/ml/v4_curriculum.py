from __future__ import annotations

import numpy as np
import pandas as pd

from .label_graph import canonicalize_pairs, positive_components
from .weak_labels import sample_weak_training, weak_confidence_weight


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


def _canonical_key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keys = canonicalize_pairs(frame[["id1", "id2", "target"]]).reset_index(drop=True)
    return pd.DataFrame({"_key1": keys["id1"], "_key2": keys["id2"]})


def _prepare_serialized_weak_rows(
    weak: pd.DataFrame,
    human: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Filter weak examples without ever mutating serialized ID/text orientation."""
    work = weak.copy().reset_index(drop=True)
    target = pd.to_numeric(work["target"], errors="raise").astype(float)
    if ((target < 0.0) | (target > 1.0)).any():
        raise ValueError("weak target must be in [0,1]")
    work["weak_weight"] = target.map(weak_confidence_weight).astype(float)
    work["hard_target"] = (target >= 0.5).astype(np.int8)
    work = work.loc[work["weak_weight"] > 0].reset_index(drop=True)
    if work.empty:
        return work

    key_columns = _canonical_key_columns(work)
    work[["_key1", "_key2"]] = key_columns[["_key1", "_key2"]]
    work["_confidence"] = (work["target"].astype(float) - 0.5).abs()
    # Deduplicate by a canonical *key* only. Keep the complete winning original
    # serialized row, including its id1/id2 and matching text_a/text_b orientation.
    work = (
        work.sort_values(
            ["_key1", "_key2", "_confidence"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(["_key1", "_key2"], keep="first")
        .reset_index(drop=True)
    )

    human_keys = _canonical_key_columns(human)
    exact_human = set(
        human_keys[["_key1", "_key2"]].itertuples(index=False, name=None)
    )
    exact_mask = np.fromiter(
        (
            (a, b) in exact_human
            for a, b in work[["_key1", "_key2"]].itertuples(index=False, name=None)
        ),
        dtype=bool,
        count=len(work),
    )
    work = work.loc[~exact_mask].reset_index(drop=True)

    components = positive_components(human[["id1", "id2", "target"]])
    false_negative: list[bool] = []
    for row in work.itertuples(index=False):
        a, b = row.id1, row.id2
        same_component = (
            a in components and b in components and components[a] == components[b]
        )
        false_negative.append(bool(row.hard_target == 0 and same_component))
    if false_negative:
        work = work.loc[~np.asarray(false_negative, dtype=bool)].reset_index(drop=True)

    validation_ids = _item_ids(validation)
    work = work.loc[
        ~work["id1"].isin(validation_ids) & ~work["id2"].isin(validation_ids)
    ].copy()
    work["source"] = "weak"
    work["sample_weight"] = work["weak_weight"].astype(float)
    return work.drop(columns=["_key1", "_key2", "_confidence"], errors="ignore").reset_index(
        drop=True
    )


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

    prepared = _prepare_serialized_weak_rows(weak, human_out, validation)
    if prepared.empty:
        raise RuntimeError("no eligible weak rows remain after leakage/conflict filtering")
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
