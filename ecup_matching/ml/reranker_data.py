from __future__ import annotations

import numbers
from typing import Mapping

import pandas as pd

from .features import normalize_items
from .textnorm import ItemNorm, clean_text


def _item_order_key(item: ItemNorm) -> tuple[int, object, str]:
    value = item.item_id
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return (0, float(value), "")
    return (1, 0.0, f"{type(value).__name__}:{value!s}")


def _attribute_weight(
    item: ItemNorm,
    key: str,
    importance: Mapping[str, Mapping[str, float]] | None,
) -> float:
    if not importance:
        return 1.0
    category_weights = importance.get(item.category, {})
    leaf = clean_text(key).rsplit(".", 1)[-1]
    return float(category_weights.get(key, category_weights.get(leaf, 1.0)))


def serialize_item_text(
    item: ItemNorm,
    attribute_importance: Mapping[str, Mapping[str, float]] | None,
    *,
    max_attrs: int = 12,
    max_chars: int = 1000,
) -> str:
    """Serialize one normalized product with important attributes first."""
    if max_attrs < 0:
        raise ValueError("max_attrs must be non-negative")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    ranked: list[tuple[float, str, str]] = []
    for key, value in item.attrs.items():
        key_norm = clean_text(key)
        value_norm = clean_text(value)
        if not key_norm or not value_norm:
            continue
        ranked.append(
            (-_attribute_weight(item, key_norm, attribute_importance), key_norm, value_norm)
        )
    ranked.sort()
    selected = ranked[:max_attrs]

    pieces = [item.name or "[missing name]"]
    if selected:
        pieces.append("[attr]")
        pieces.extend(f"{key}={value}" for _, key, value in selected)
    text = " ; ".join(pieces)
    return text[:max_chars].rstrip()


def serialize_pair(
    a: ItemNorm,
    b: ItemNorm,
    attribute_importance: Mapping[str, Mapping[str, float]] | None,
    *,
    max_attrs: int = 12,
    max_chars: int = 1000,
) -> tuple[str, str]:
    """Return a canonical A/B text pair independent of pair direction."""
    left, right = sorted((a, b), key=_item_order_key)
    return (
        serialize_item_text(
            left,
            attribute_importance,
            max_attrs=max_attrs,
            max_chars=max_chars,
        ),
        serialize_item_text(
            right,
            attribute_importance,
            max_attrs=max_attrs,
            max_chars=max_chars,
        ),
    )


def build_reranker_examples(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    attribute_importance: Mapping[str, Mapping[str, float]] | None,
    *,
    max_attrs: int = 12,
    max_chars: int = 1000,
) -> pd.DataFrame:
    """Materialize pair texts while preserving soft targets, weights and provenance."""
    missing = {"id1", "id2", "target"} - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs missing required columns: {sorted(missing)}")
    cache = normalize_items(items)

    rows: list[dict[str, object]] = []
    for row in pairs.itertuples(index=False):
        id1, id2 = row.id1, row.id2
        if id1 not in cache or id2 not in cache:
            raise KeyError(f"pair references missing item: {id1!r}, {id2!r}")
        text_a, text_b = serialize_pair(
            cache[id1],
            cache[id2],
            attribute_importance,
            max_attrs=max_attrs,
            max_chars=max_chars,
        )
        category = getattr(row, "category", cache[id1].category or cache[id2].category)
        sample_weight = float(getattr(row, "sample_weight", 1.0))
        source = str(getattr(row, "source", "human"))
        rows.append(
            {
                "id1": id1,
                "id2": id2,
                "target": float(row.target),
                "sample_weight": sample_weight,
                "source": source,
                "category": str(category),
                "text_a": text_a,
                "text_b": text_b,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "id1",
            "id2",
            "target",
            "sample_weight",
            "source",
            "category",
            "text_a",
            "text_b",
        ],
    )
