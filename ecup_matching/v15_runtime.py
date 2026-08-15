"""Submission-time data/runtime helpers for E-CUP v15."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd


def referenced_item_ids(pairs: pd.DataFrame) -> set:
    required = {"id1", "id2"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"missing pair columns: {sorted(missing)}")
    return set(pairs["id1"].tolist()) | set(pairs["id2"].tolist())


def materialize_referenced_items(items: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    if "id" not in items.columns:
        raise ValueError("items must contain id")
    ids = referenced_item_ids(pairs)
    subset = items.loc[items["id"].isin(ids)].copy()
    found = set(subset["id"].tolist())
    missing = ids - found
    if missing:
        raise ValueError(f"missing referenced items: {len(missing)}")
    return subset.reset_index(drop=True)


def build_normalized_item_cache(items: pd.DataFrame, normalizer) -> dict:
    required = {"id", "name", "attributes", "category"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"missing item columns: {sorted(missing)}")
    cache = {}
    for row in items.itertuples(index=False):
        item_id = getattr(row, "id")
        if item_id not in cache:
            cache[item_id] = normalizer(
                getattr(row, "name"),
                getattr(row, "attributes"),
                getattr(row, "category"),
            )
    return cache


def write_predictions(pairs: pd.DataFrame, predictions: Iterable[float], output_path: str | Path) -> None:
    preds = np.asarray(list(predictions), dtype=float)
    if len(preds) != len(pairs):
        raise ValueError(f"prediction row count mismatch: {len(preds)} != {len(pairs)}")
    if not np.isfinite(preds).all():
        raise ValueError("predictions must be finite")
    out = pairs.loc[:, ["id1", "id2"]].copy()
    out["predict"] = preds
    out.to_csv(output_path, index=False)
