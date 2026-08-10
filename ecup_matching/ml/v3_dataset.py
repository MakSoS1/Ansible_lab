from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd


PAIR_COLUMNS = ("id1", "id2")


def _item_ids(frame: pd.DataFrame) -> set[object]:
    missing = [column for column in PAIR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing pair columns: {missing}")
    return set(frame["id1"].dropna()) | set(frame["id2"].dropna())


def assert_no_item_overlap(train_pairs: pd.DataFrame, validation_pairs: pd.DataFrame) -> None:
    overlap = _item_ids(train_pairs) & _item_ids(validation_pairs)
    if overlap:
        raise RuntimeError(f"found {len(overlap)} overlapping item IDs")


def _without_validation_items(
    frame: pd.DataFrame, validation_item_ids: set[object]
) -> pd.DataFrame:
    if not validation_item_ids:
        return frame.copy().reset_index(drop=True)
    keep = ~frame["id1"].isin(validation_item_ids) & ~frame["id2"].isin(validation_item_ids)
    return frame.loc[keep].copy().reset_index(drop=True)


def _sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    if len(frame) <= count:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=count, random_state=seed).reset_index(drop=True)


def sample_v3_training_pairs(
    human_pairs: pd.DataFrame,
    weak_pairs: pd.DataFrame,
    *,
    validation_item_ids: Iterable[object],
    max_rows: int,
    priority_categories: set[str],
    priority_fraction: float = 0.60,
    seed: int = 2026,
) -> pd.DataFrame:
    """Build a deterministic compact v3 pair sample without validation leakage.

    Human positives are mandatory. The remaining budget is filled first from
    configured priority categories and then from the rest of the available
    human-negative/weak pool. If one pool is short, the other fills the gap.
    """
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if not 0.0 <= priority_fraction <= 1.0:
        raise ValueError("priority_fraction must be between 0 and 1")

    validation_ids = set(validation_item_ids)
    human = _without_validation_items(human_pairs, validation_ids)
    weak = _without_validation_items(weak_pairs, validation_ids)

    required_human = {"id1", "id2", "target", "category"}
    required_weak = required_human | {"weak_weight"}
    missing_human = required_human - set(human.columns)
    missing_weak = required_weak - set(weak.columns)
    if missing_human:
        raise ValueError(f"human_pairs missing columns: {sorted(missing_human)}")
    if missing_weak:
        raise ValueError(f"weak_pairs missing columns: {sorted(missing_weak)}")

    human = human.copy()
    weak = weak.copy()
    human["source"] = "human"
    weak["source"] = "weak"
    if "weak_weight" not in human.columns:
        human["weak_weight"] = 1.0

    columns = ["id1", "id2", "target", "category", "weak_weight", "source"]
    human = human[columns]
    weak = weak[columns]

    mandatory = human[human["target"].astype(float) >= 0.5].copy().reset_index(drop=True)
    if len(mandatory) > max_rows:
        raise ValueError(
            f"max_rows={max_rows} cannot preserve {len(mandatory)} mandatory human positives"
        )

    human_negative = human[human["target"].astype(float) < 0.5].copy()
    discretionary = pd.concat([human_negative, weak], ignore_index=True)
    budget = max_rows - len(mandatory)
    if len(discretionary) < budget:
        raise ValueError(
            f"not enough leakage-safe rows to fill max_rows={max_rows}: "
            f"mandatory={len(mandatory)}, discretionary={len(discretionary)}"
        )

    priority_mask = discretionary["category"].astype(str).isin(priority_categories)
    priority_pool = discretionary.loc[priority_mask].reset_index(drop=True)
    regular_pool = discretionary.loc[~priority_mask].reset_index(drop=True)

    requested_priority = min(budget, math.ceil(budget * priority_fraction))
    priority_count = min(requested_priority, len(priority_pool))
    selected_priority = _sample(priority_pool, priority_count, seed)

    remaining = budget - len(selected_priority)
    selected_regular = _sample(regular_pool, min(remaining, len(regular_pool)), seed + 1)
    remaining -= len(selected_regular)

    if remaining:
        used_priority = set(selected_priority.index)
        if len(priority_pool) > len(selected_priority):
            # Re-sample from the complete priority pool with a distinct seed,
            # excluding already selected pair identities rather than relying on
            # transient DataFrame indexes.
            selected_keys = set(
                map(tuple, selected_priority[["id1", "id2", "target", "category"]].to_numpy())
            )
            leftover_priority = priority_pool[
                ~priority_pool[["id1", "id2", "target", "category"]]
                .apply(tuple, axis=1)
                .isin(selected_keys)
            ].reset_index(drop=True)
            fill = _sample(leftover_priority, remaining, seed + 2)
            selected_priority = pd.concat([selected_priority, fill], ignore_index=True)
            remaining -= len(fill)

    if remaining:
        selected_keys = set(
            map(tuple, selected_regular[["id1", "id2", "target", "category"]].to_numpy())
        )
        leftover_regular = regular_pool[
            ~regular_pool[["id1", "id2", "target", "category"]]
            .apply(tuple, axis=1)
            .isin(selected_keys)
        ].reset_index(drop=True)
        fill = _sample(leftover_regular, remaining, seed + 3)
        selected_regular = pd.concat([selected_regular, fill], ignore_index=True)
        remaining -= len(fill)

    if remaining:
        raise RuntimeError(f"failed to fill v3 sample budget; {remaining} rows missing")

    out = pd.concat([mandatory, selected_priority, selected_regular], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed + 4).reset_index(drop=True)
    if len(out) != max_rows:
        raise RuntimeError(f"expected {max_rows} sampled rows, got {len(out)}")
    if validation_ids & _item_ids(out):
        raise RuntimeError("sample contains validation item IDs")
    return out
