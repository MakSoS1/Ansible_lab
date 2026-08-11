from __future__ import annotations

import pandas as pd


def forbidden_weak_item_ids(
    matches: pd.DataFrame,
    manifest: dict,
    *,
    held_fold: int,
) -> set[object]:
    """Return gold + held-fold items that weak training must never touch."""
    if not {"id1", "id2"}.issubset(matches.columns):
        raise ValueError("matches must contain id1 and id2")
    folds = manifest.get("fold_rows", [])
    if held_fold < 0 or held_fold >= len(folds):
        raise ValueError(f"unknown held fold {held_fold}")
    gold_rows = [int(row) for row in manifest.get("gold_rows", [])]
    held_rows = [int(row) for row in folds[held_fold]]
    rows = gold_rows + held_rows
    if any(row < 0 or row >= len(matches) for row in rows):
        raise IndexError("manifest contains out-of-range row")
    subset = matches.iloc[rows]
    return set(subset["id1"].tolist()) | set(subset["id2"].tolist())
