from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def pseudo_binary_labels(target, *, low: float = 0.05, high: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Convert only extreme soft labels into diagnostic pseudo-binary labels.

    Values exactly on the thresholds are deliberately treated as ambiguous.
    The returned labels correspond only to rows where ``mask`` is True.
    """
    if not (0.0 <= float(low) < float(high) <= 1.0):
        raise ValueError("expected 0 <= low < high <= 1")
    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("target must be a finite one-dimensional array")
    mask = (values < float(low)) | (values > float(high))
    labels = (values[mask] > float(high)).astype(np.int8)
    return np.asarray(mask, dtype=bool), labels


def build_testlike_slice(
    llm_pairs: pd.DataFrame,
    human_item_ids: set[object],
    *,
    max_rows: int,
    seed: int = 2026,
) -> pd.DataFrame:
    required = {"id1", "id2", "target"}
    missing = required - set(llm_pairs.columns)
    if missing:
        raise ValueError(f"LLM pairs missing columns: {sorted(missing)}")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    eligible = llm_pairs.loc[
        ~llm_pairs["id1"].isin(human_item_ids)
        & ~llm_pairs["id2"].isin(human_item_ids)
    ].copy()
    if len(eligible) < max_rows:
        raise ValueError(f"available test-like rows {len(eligible)} < requested {max_rows}")
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(len(eligible), size=int(max_rows), replace=False))
    out = eligible.iloc[chosen].reset_index(drop=True)
    if (set(out["id1"].tolist()) | set(out["id2"].tolist())) & human_item_ids:
        raise RuntimeError("human item leaked into test-like slice")
    out.attrs["diagnostic_only"] = True
    out.attrs["selection_seed"] = int(seed)
    out.attrs["source"] = "human-item-excluded-llm-candidates"
    return out


def pseudo_macro_ap_report(
    frame: pd.DataFrame,
    scores,
    *,
    low: float = 0.05,
    high: float = 0.95,
    category_col: str = "category",
    target_col: str = "target",
) -> dict[str, object]:
    required = {category_col, target_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    score = np.asarray(scores, dtype=np.float64)
    if score.ndim != 1 or len(score) != len(frame) or not np.isfinite(score).all():
        raise ValueError("scores must be finite and aligned")
    mask, labels = pseudo_binary_labels(frame[target_col].to_numpy(float), low=low, high=high)
    work = frame.loc[mask, [category_col]].copy().reset_index(drop=True)
    work["_label"] = labels
    kept_score = score[mask]
    per_category: dict[str, float] = {}
    for category, idx in work.groupby(category_col, sort=True).indices.items():
        pos = np.asarray(idx, dtype=np.int64)
        y = work.iloc[pos]["_label"].to_numpy(np.int8)
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError(f"category {category!r} must contain both pseudo classes")
        per_category[str(category)] = float(average_precision_score(y, kept_score[pos]))
    if not per_category:
        raise ValueError("no categories remain after pseudo-label filtering")
    return {
        "diagnostic_only": True,
        "pseudo_label_thresholds": {"low": float(low), "high": float(high)},
        "pseudo_label_rows": int(mask.sum()),
        "source_rows": int(len(frame)),
        "categories": int(len(per_category)),
        "macro_pseudo_average_precision": float(np.mean(list(per_category.values()))),
        "per_category_pseudo_ap": per_category,
        "true_test_ap_claimed": False,
    }


def _rank_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def soft_rank_report(
    frame: pd.DataFrame,
    scores,
    *,
    category_col: str = "category",
    target_col: str = "target",
) -> dict[str, object]:
    required = {category_col, target_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    score = np.asarray(scores, dtype=np.float64)
    target = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=np.float64)
    if score.ndim != 1 or len(score) != len(frame) or not np.isfinite(score).all():
        raise ValueError("scores must be finite and aligned")
    if not np.isfinite(target).all():
        raise ValueError("target contains nonfinite values")
    per_category: dict[str, float] = {}
    for category, idx in frame.groupby(category_col, sort=True).indices.items():
        pos = np.asarray(idx, dtype=np.int64)
        if len(pos) < 2:
            raise ValueError(f"category {category!r} needs at least two rows")
        a = _rank_average(score[pos])
        b = _rank_average(target[pos])
        if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            raise ValueError(f"category {category!r} has constant score or soft target")
        per_category[str(category)] = float(np.corrcoef(a, b)[0, 1])
    return {
        "diagnostic_only": True,
        "rows": int(len(frame)),
        "categories": int(len(per_category)),
        "macro_spearman": float(np.mean(list(per_category.values()))),
        "per_category_spearman": per_category,
        "true_test_ap_claimed": False,
    }


__all__ = [
    "build_testlike_slice",
    "pseudo_binary_labels",
    "pseudo_macro_ap_report",
    "soft_rank_report",
]
