"""Leakage-safe validation helpers for E-CUP v15."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class MacroAPResult:
    macro_ap: float
    per_category: dict[str, float]


def compute_macro_ap(
    frame: pd.DataFrame,
    *,
    expected_categories: Iterable[str] | None = None,
    category_col: str = "category",
    target_col: str = "target",
    predict_col: str = "predict",
) -> MacroAPResult:
    required = {category_col, target_col, predict_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing metric columns: {sorted(missing)}")
    if not np.isfinite(frame[predict_col].to_numpy(dtype=float)).all():
        raise ValueError("predictions must be finite")

    if expected_categories is None:
        categories = sorted(frame[category_col].dropna().astype(str).unique().tolist())
    else:
        categories = [str(c) for c in expected_categories]
        observed = set(frame[category_col].astype(str).unique())
        absent = [c for c in categories if c not in observed]
        if absent:
            raise ValueError(f"missing expected categories: {absent}")

    if not categories:
        raise ValueError("no categories to score")

    per_category: dict[str, float] = {}
    cat_as_str = frame[category_col].astype(str)
    for category in categories:
        part = frame.loc[cat_as_str == category]
        if part.empty:
            raise ValueError(f"empty category: {category}")
        y_true = part[target_col].to_numpy()
        y_score = part[predict_col].to_numpy(dtype=float)
        per_category[category] = float(average_precision_score(y_true, y_score))

    return MacroAPResult(
        macro_ap=float(np.mean(list(per_category.values()))),
        per_category=per_category,
    )


def validate_oof_integrity(
    frame: pd.DataFrame,
    *,
    expected_row_indexes: set[int] | Iterable[int],
    row_index_col: str = "row_index",
    predict_col: str = "predict",
) -> None:
    if row_index_col not in frame.columns:
        raise ValueError(f"missing OOF row index column: {row_index_col}")
    if predict_col not in frame.columns:
        raise ValueError(f"missing OOF prediction column: {predict_col}")
    if frame[row_index_col].duplicated().any():
        duplicates = frame.loc[frame[row_index_col].duplicated(keep=False), row_index_col].tolist()[:10]
        raise ValueError(f"duplicate OOF row indexes: {duplicates}")
    expected = {int(x) for x in expected_row_indexes}
    observed = {int(x) for x in frame[row_index_col].tolist()}
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        extra = sorted(observed - expected)[:10]
        raise ValueError(f"OOF coverage mismatch: missing={missing} extra={extra}")
    if not np.isfinite(frame[predict_col].to_numpy(dtype=float)).all():
        raise ValueError("OOF predictions must be finite")


def assert_item_disjoint(train_item_ids: Iterable[int], held_item_ids: Iterable[int]) -> None:
    overlap = set(train_item_ids) & set(held_item_ids)
    if overlap:
        raise ValueError(f"train/held item overlap detected: {len(overlap)} items")
