from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import pandas as pd

from .textnorm import ItemNorm, clean_text, normalize_item


_BRAND_ALIASES = {
    "brand",
    "бренд",
    "manufacturer",
    "производитель",
    "марка",
    "vendor",
}


def _leaf_key(key: str) -> str:
    return clean_text(str(key)).rsplit(".", 1)[-1]


def brand_from_item(item: ItemNorm) -> str | None:
    """Return the first deterministic normalized brand/manufacturer attribute."""
    candidates: list[tuple[str, str]] = []
    for key, value in item.attrs.items():
        if _leaf_key(key) in _BRAND_ALIASES:
            normalized = clean_text(value)
            if normalized:
                candidates.append((clean_text(key), normalized))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _category_weights(
    category: str,
    importance: Mapping[str, Mapping[str, float]] | None,
) -> Mapping[str, float]:
    if not importance:
        return {}
    return importance.get(category, importance.get(clean_text(category), {}))


def weighted_attribute_stats(
    a: ItemNorm,
    b: ItemNorm,
    importance: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, float]:
    """Compare shared attributes using category-specific importance weights."""
    category = a.category or b.category
    weights = _category_weights(category, importance)
    shared = set(a.attrs) & set(b.attrs)

    agreement_weight = 0.0
    conflict_weight = 0.0
    total = 0.0
    for key in shared:
        leaf = _leaf_key(key)
        weight = float(weights.get(key, weights.get(leaf, 1.0)))
        weight = max(weight, 0.0)
        total += weight
        if a.attrs[key] == b.attrs[key]:
            agreement_weight += weight
        else:
            conflict_weight += weight

    brand_a = brand_from_item(a)
    brand_b = brand_from_item(b)
    comparable_brand = bool(brand_a and brand_b)
    return {
        "weighted_attr_agreement": float(agreement_weight / total) if total > 0 else 0.0,
        "weighted_attr_conflict": float(conflict_weight / total) if total > 0 else 0.0,
        "brand_match": float(comparable_brand and brand_a == brand_b),
        "brand_conflict": float(comparable_brand and brand_a != brand_b),
    }


def learn_attribute_importance(
    items: pd.DataFrame,
    train_pairs: pd.DataFrame,
    min_support: int = 20,
) -> dict[str, dict[str, float]]:
    """Learn simple label-consistency importance from training pairs only.

    An attribute is useful for identity when positives tend to agree on it and
    negatives tend to disagree. The returned score is bounded away from zero
    and normalized to mean ~1 within each category, making it suitable for
    deterministic pair features and serialization in a model manifest.
    """
    if min_support <= 0:
        raise ValueError("min_support must be positive")
    required_items = {"id", "name", "attributes", "category"}
    missing_items = required_items - set(items.columns)
    if missing_items:
        raise ValueError(f"items missing required columns: {sorted(missing_items)}")
    missing_pairs = {"id1", "id2", "target"} - set(train_pairs.columns)
    if missing_pairs:
        raise ValueError(f"train_pairs missing required columns: {sorted(missing_pairs)}")

    item_index: dict[object, ItemNorm] = {}
    for row in items[["id", "name", "attributes", "category"]].itertuples(index=False, name=None):
        item_index[row[0]] = normalize_item(*row)

    stats: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"pos_equal": 0.0, "pos_total": 0.0, "neg_diff": 0.0, "neg_total": 0.0, "support": 0.0})
    )
    for id1, id2, target in train_pairs[["id1", "id2", "target"]].itertuples(index=False, name=None):
        if id1 not in item_index or id2 not in item_index:
            continue
        a, b = item_index[id1], item_index[id2]
        category = a.category or b.category
        for key in set(a.attrs) & set(b.attrs):
            equal = a.attrs[key] == b.attrs[key]
            entry = stats[category][_leaf_key(key)]
            entry["support"] += 1.0
            if float(target) >= 0.5:
                entry["pos_total"] += 1.0
                entry["pos_equal"] += float(equal)
            else:
                entry["neg_total"] += 1.0
                entry["neg_diff"] += float(not equal)

    result: dict[str, dict[str, float]] = {}
    for category, by_key in stats.items():
        raw: dict[str, float] = {}
        for key, entry in by_key.items():
            if entry["support"] < min_support:
                continue
            pos_consistency = (
                entry["pos_equal"] / entry["pos_total"] if entry["pos_total"] else 0.5
            )
            neg_consistency = (
                entry["neg_diff"] / entry["neg_total"] if entry["neg_total"] else 0.5
            )
            # 0.05 prevents supported but weak attributes from disappearing;
            # squaring rewards attributes consistent in both directions.
            score = 0.05 + ((pos_consistency + neg_consistency) / 2.0) ** 2
            raw[key] = float(score)
        if raw:
            mean = sum(raw.values()) / len(raw)
            result[category] = {key: float(value / mean) for key, value in sorted(raw.items())}
    return result
