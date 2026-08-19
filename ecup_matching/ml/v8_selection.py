from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import pandas as pd


def _anchor_key(seed: int, category: str, anchor: object) -> bytes:
    payload = f"{int(seed)}\x1f{category}\x1f{type(anchor).__name__}\x1f{repr(anchor)}".encode(
        "utf-8", errors="surrogatepass"
    )
    return hashlib.sha256(payload).digest()


def anchor_disjoint_split(
    frame: pd.DataFrame,
    *,
    seed: int = 2026,
    tune_fraction: float = 0.5,
    anchor_col: str = "id1",
    category_col: str = "category",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Deterministically split complete retrieval anchors within each category.

    No target or score column is inspected.  Every row for a selected anchor is
    assigned to the same side, so graph candidate lists remain intact.
    """
    required = {anchor_col, category_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"selection frame missing columns: {sorted(missing)}")
    if not 0.0 < float(tune_fraction) < 1.0:
        raise ValueError("tune_fraction must be strictly between 0 and 1")
    if len(frame) == 0:
        raise ValueError("selection frame is empty")

    anchors = frame[[anchor_col, category_col]].copy()
    anchors[category_col] = anchors[category_col].astype(str)
    per_anchor_categories = anchors.groupby(anchor_col, sort=False, dropna=False)[category_col].nunique()
    bad = per_anchor_categories[per_anchor_categories > 1]
    if len(bad):
        raise ValueError("retrieval anchor appears in multiple categories")

    unique = anchors.drop_duplicates(anchor_col, keep="first")
    tune_anchors: set[object] = set()
    confirm_anchors: set[object] = set()
    per_category: dict[str, dict[str, int]] = {}

    for category, group in unique.groupby(category_col, sort=True, dropna=False):
        values = group[anchor_col].tolist()
        n = len(values)
        if n < 2:
            raise ValueError(f"category {category!r} needs at least two retrieval anchors")
        ordered = sorted(values, key=lambda anchor: _anchor_key(seed, str(category), anchor))
        n_tune = int(round(n * float(tune_fraction)))
        n_tune = min(n - 1, max(1, n_tune))
        chosen = set(ordered[:n_tune])
        held = set(ordered[n_tune:])
        tune_anchors.update(chosen)
        confirm_anchors.update(held)
        per_category[str(category)] = {
            "anchors": int(n),
            "tune_anchors": int(len(chosen)),
            "confirm_anchors": int(len(held)),
        }

    overlap = tune_anchors & confirm_anchors
    if overlap:
        raise RuntimeError("anchor split produced overlap")
    tune = frame[anchor_col].isin(tune_anchors).to_numpy(dtype=bool)
    confirm = frame[anchor_col].isin(confirm_anchors).to_numpy(dtype=bool)
    if np.any(tune & confirm) or not np.all(tune | confirm):
        raise RuntimeError("anchor split row masks are not disjoint/exhaustive")

    meta: dict[str, object] = {
        "seed": int(seed),
        "tune_fraction": float(tune_fraction),
        "tune_rows": int(tune.sum()),
        "confirm_rows": int(confirm.sum()),
        "tune_anchors": int(len(tune_anchors)),
        "confirm_anchors": int(len(confirm_anchors)),
        "anchor_overlap": int(len(overlap)),
        "per_category": per_category,
        "target_free": True,
    }
    return tune, confirm, meta


def category_rank_percentile(
    frame: pd.DataFrame,
    scores,
    *,
    category_col: str = "category",
) -> np.ndarray:
    if category_col not in frame.columns:
        raise ValueError(f"frame missing category column {category_col!r}")
    score = np.asarray(scores, dtype=np.float64)
    if score.ndim != 1 or len(score) != len(frame) or not np.isfinite(score).all():
        raise ValueError("scores must be finite, one-dimensional and aligned")
    if len(score) == 0:
        raise ValueError("scores are empty")
    categories = frame[category_col].astype(str).reset_index(drop=True)
    values = pd.Series(score)
    count = categories.groupby(categories, sort=False).transform("size").to_numpy(dtype=np.float64)
    rank = values.groupby(categories, sort=False).rank(method="average", ascending=True).to_numpy(dtype=np.float64)
    denom = np.maximum(1.0, count - 1.0)
    percentile = np.where(count <= 1.0, 1.0, (rank - 1.0) / denom)
    percentile = np.clip(percentile, 0.0, 1.0)
    if not np.isfinite(percentile).all():
        raise RuntimeError("category rank percentile produced nonfinite values")
    return percentile.astype(np.float64, copy=False)


def rank_blend(
    frame: pd.DataFrame,
    scores: Mapping[str, object],
    weights: Mapping[str, float],
    *,
    category_col: str = "category",
) -> np.ndarray:
    if not scores:
        raise ValueError("rank blend needs at least one score source")
    if set(scores) != set(weights):
        raise ValueError("score and weight keys must match exactly")
    ordered_keys = sorted(scores)
    w = np.asarray([float(weights[key]) for key in ordered_keys], dtype=np.float64)
    if not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("blend weights must be finite and nonnegative")
    if not np.isclose(float(w.sum()), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("blend weights must sum to one")
    out = np.zeros(len(frame), dtype=np.float64)
    for key, weight in zip(ordered_keys, w.tolist()):
        if weight == 0.0:
            continue
        out += weight * category_rank_percentile(frame, scores[key], category_col=category_col)
    if not np.isfinite(out).all():
        raise RuntimeError("rank blend produced nonfinite values")
    return out


__all__ = ["anchor_disjoint_split", "category_rank_percentile", "rank_blend"]
