from __future__ import annotations

from collections import defaultdict
import math
from typing import Mapping, Any

import numpy as np
import pandas as pd

from .category_attrs import _leaf_key
from .features import normalize_items
from .textnorm import ItemNorm


ATTRIBUTE_EVIDENCE_FEATURES: tuple[str, ...] = (
    "attr_evidence_sum",
    "attr_evidence_mean",
    "attr_evidence_positive",
    "attr_evidence_negative",
    "attr_evidence_min",
    "attr_evidence_max",
    "attr_evidence_count",
)


def fit_attribute_evidence(
    items: pd.DataFrame,
    train_pairs: pd.DataFrame,
    *,
    min_support: int = 20,
    smoothing: float = 2.0,
    item_cache: Mapping[object, ItemNorm] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Learn category/key-specific equal-vs-different likelihood ratios from train only."""
    if min_support <= 0:
        raise ValueError("min_support must be positive")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    missing = {"id1", "id2", "target"} - set(train_pairs.columns)
    if missing:
        raise ValueError(f"train_pairs missing columns: {sorted(missing)}")
    cache = dict(item_cache) if item_cache is not None else normalize_items(items)

    stats: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "pos_equal": 0.0,
                "pos_diff": 0.0,
                "neg_equal": 0.0,
                "neg_diff": 0.0,
                "support": 0.0,
            }
        )
    )
    for id1, id2, raw_target in train_pairs[["id1", "id2", "target"]].itertuples(index=False, name=None):
        if id1 not in cache or id2 not in cache:
            continue
        a, b = cache[id1], cache[id2]
        category = str(a.category or b.category or "__missing__")
        positive = float(raw_target) >= 0.5
        for key in set(a.attrs) & set(b.attrs):
            leaf = _leaf_key(key)
            equal = a.attrs[key] == b.attrs[key]
            entry = stats[category][leaf]
            entry["support"] += 1.0
            if positive and equal:
                entry["pos_equal"] += 1.0
            elif positive:
                entry["pos_diff"] += 1.0
            elif equal:
                entry["neg_equal"] += 1.0
            else:
                entry["neg_diff"] += 1.0

    result: dict[str, dict[str, dict[str, float]]] = {}
    for category, by_key in stats.items():
        output: dict[str, dict[str, float]] = {}
        for key, entry in by_key.items():
            if entry["support"] < min_support:
                continue
            pos_total = entry["pos_equal"] + entry["pos_diff"]
            neg_total = entry["neg_equal"] + entry["neg_diff"]
            pos_equal = (entry["pos_equal"] + smoothing) / (pos_total + 2.0 * smoothing)
            neg_equal = (entry["neg_equal"] + smoothing) / (neg_total + 2.0 * smoothing)
            pos_diff = (entry["pos_diff"] + smoothing) / (pos_total + 2.0 * smoothing)
            neg_diff = (entry["neg_diff"] + smoothing) / (neg_total + 2.0 * smoothing)
            output[key] = {
                "equal": float(math.log(pos_equal / neg_equal)),
                "different": float(math.log(pos_diff / neg_diff)),
                "support": float(entry["support"]),
            }
        if output:
            result[category] = dict(sorted(output.items()))
    return result


def build_attribute_evidence_features(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    evidence: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    item_cache: Mapping[object, ItemNorm] | None = None,
) -> pd.DataFrame:
    if not {"id1", "id2"}.issubset(pairs.columns):
        raise ValueError("pairs must contain id1 and id2")
    cache = dict(item_cache) if item_cache is not None else normalize_items(items)
    rows: list[dict[str, float]] = []
    for id1, id2 in pairs[["id1", "id2"]].itertuples(index=False, name=None):
        if id1 not in cache or id2 not in cache:
            missing = id1 if id1 not in cache else id2
            raise KeyError(f"pair references missing item {missing!r}")
        a, b = cache[id1], cache[id2]
        category = str(a.category or b.category or "__missing__")
        category_evidence = evidence.get(category, {})
        values: list[float] = []
        for key in set(a.attrs) & set(b.attrs):
            learned = category_evidence.get(_leaf_key(key))
            if not learned:
                continue
            state = "equal" if a.attrs[key] == b.attrs[key] else "different"
            values.append(float(learned[state]))

        if values:
            arr = np.asarray(values, dtype=np.float64)
            positive = float(arr[arr > 0].sum()) if np.any(arr > 0) else 0.0
            negative = float(arr[arr < 0].sum()) if np.any(arr < 0) else 0.0
            row = {
                "attr_evidence_sum": float(arr.sum()),
                "attr_evidence_mean": float(arr.mean()),
                "attr_evidence_positive": positive,
                "attr_evidence_negative": negative,
                "attr_evidence_min": float(arr.min()),
                "attr_evidence_max": float(arr.max()),
                "attr_evidence_count": float(len(arr)),
            }
        else:
            row = {name: 0.0 for name in ATTRIBUTE_EVIDENCE_FEATURES}
        rows.append(row)
    return pd.DataFrame(rows, columns=ATTRIBUTE_EVIDENCE_FEATURES).astype(np.float32)
