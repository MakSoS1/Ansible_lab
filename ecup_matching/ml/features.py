from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable

import numpy as np
import pandas as pd

from .textnorm import ItemNorm, normalize_item


FEATURE_NAMES: tuple[str, ...] = (
    "category",
    "same_category",
    "name_exact",
    "name_contains",
    "fuzz_ratio",
    "fuzz_partial_ratio",
    "fuzz_token_sort",
    "fuzz_token_set",
    "name_token_jaccard",
    "name_char3_jaccard",
    "name_len_ratio",
    "name_len_diff",
    "number_jaccard",
    "number_conflict",
    "model_code_jaccard",
    "model_code_conflict",
    "quantity_jaccard",
    "quantity_conflict",
    "attr_key_jaccard",
    "attr_shared_keys",
    "attr_value_agreement",
    "attr_value_conflict",
    "attr_value_token_jaccard",
    "name_missing_any",
    "attr_missing_any",
)


def _jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else 1.0


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b, autojunk=False).ratio())


def _partial_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if short in long:
        return 1.0
    matcher = SequenceMatcher(None, short, long, autojunk=False)
    block = matcher.find_longest_match(0, len(short), 0, len(long))
    return float(block.size / max(1, len(short)))


def _set_conflict(a: frozenset, b: frozenset) -> float:
    return float(bool(a) and bool(b) and not bool(a & b))


def _quantity_conflict(a: frozenset[tuple[str, float]], b: frozenset[tuple[str, float]]) -> float:
    if not a or not b:
        return 0.0
    by_dim_a: dict[str, set[float]] = {}
    by_dim_b: dict[str, set[float]] = {}
    for dim, value in a:
        by_dim_a.setdefault(dim, set()).add(value)
    for dim, value in b:
        by_dim_b.setdefault(dim, set()).add(value)
    common = set(by_dim_a) & set(by_dim_b)
    if not common:
        return 0.0
    for dim in common:
        if by_dim_a[dim].isdisjoint(by_dim_b[dim]):
            return 1.0
    return 0.0


def _attr_stats(a: ItemNorm, b: ItemNorm) -> tuple[float, float, float, float, float]:
    keys_a, keys_b = set(a.attrs), set(b.attrs)
    shared = keys_a & keys_b
    key_jaccard = _jaccard(keys_a, keys_b)
    shared_ratio = float(len(shared) / max(1, min(len(keys_a), len(keys_b)))) if (keys_a and keys_b) else 0.0
    if not shared:
        agreement = 0.0
        conflict = 0.0
    else:
        equal = sum(a.attrs[k] == b.attrs[k] for k in shared)
        agreement = float(equal / len(shared))
        conflict = float(1.0 - agreement)
    value_jaccard = _jaccard(a.attr_tokens, b.attr_tokens)
    return key_jaccard, shared_ratio, agreement, conflict, value_jaccard


def _pair_features(a: ItemNorm, b: ItemNorm) -> dict[str, object]:
    name_a, name_b = a.name, b.name
    sorted_a = " ".join(sorted(a.name_tokens))
    sorted_b = " ".join(sorted(b.name_tokens))
    common_tokens = sorted(a.name_tokens & b.name_tokens)
    only_a = sorted(a.name_tokens - b.name_tokens)
    only_b = sorted(b.name_tokens - a.name_tokens)
    token_set_a = " ".join(common_tokens + only_a)
    token_set_b = " ".join(common_tokens + only_b)

    max_len = max(len(name_a), len(name_b), 1)
    min_len = min(len(name_a), len(name_b))
    attr_key_j, attr_shared, attr_agree, attr_conflict, attr_token_j = _attr_stats(a, b)

    return {
        "category": a.category or b.category or "__missing__",
        "same_category": float(a.category == b.category),
        "name_exact": float(bool(name_a) and name_a == name_b),
        "name_contains": float(bool(name_a) and bool(name_b) and (name_a in name_b or name_b in name_a)),
        "fuzz_ratio": _ratio(name_a, name_b),
        "fuzz_partial_ratio": _partial_ratio(name_a, name_b),
        "fuzz_token_sort": _ratio(sorted_a, sorted_b),
        "fuzz_token_set": _ratio(token_set_a, token_set_b),
        "name_token_jaccard": _jaccard(a.name_tokens, b.name_tokens),
        "name_char3_jaccard": _jaccard(a.name_char3, b.name_char3),
        "name_len_ratio": float(min_len / max_len),
        "name_len_diff": float(abs(len(name_a) - len(name_b)) / max_len),
        "number_jaccard": _jaccard(a.numbers, b.numbers),
        "number_conflict": _set_conflict(a.numbers, b.numbers),
        "model_code_jaccard": _jaccard(a.model_codes, b.model_codes),
        "model_code_conflict": _set_conflict(a.model_codes, b.model_codes),
        "quantity_jaccard": _jaccard(a.quantities, b.quantities),
        "quantity_conflict": _quantity_conflict(a.quantities, b.quantities),
        "attr_key_jaccard": attr_key_j,
        "attr_shared_keys": attr_shared,
        "attr_value_agreement": attr_agree,
        "attr_value_conflict": attr_conflict,
        "attr_value_token_jaccard": attr_token_j,
        "name_missing_any": float(not name_a or not name_b),
        "attr_missing_any": float(not a.attrs or not b.attrs),
    }


def normalize_items(items: pd.DataFrame) -> dict[object, ItemNorm]:
    required = {"id", "name", "attributes", "category"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items is missing required columns: {sorted(missing)}")
    result: dict[object, ItemNorm] = {}
    for row in items[["id", "name", "attributes", "category"]].itertuples(index=False, name=None):
        item_id, name, attributes, category = row
        result[item_id] = normalize_item(item_id, name, attributes, category)
    return result


def build_pair_features(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    item_cache: dict[object, ItemNorm] | None = None,
) -> pd.DataFrame:
    """Build deterministic pair features in input order, parsing each item at most once."""
    if not {"id1", "id2"}.issubset(pairs.columns):
        raise ValueError("pairs must contain id1 and id2")
    cache = item_cache if item_cache is not None else normalize_items(items)
    rows: list[dict[str, object]] = []
    for id1, id2 in pairs[["id1", "id2"]].itertuples(index=False, name=None):
        if id1 not in cache or id2 not in cache:
            raise KeyError(f"pair references missing item: {id1!r}, {id2!r}")
        rows.append(_pair_features(cache[id1], cache[id2]))
    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    numeric = [c for c in FEATURE_NAMES if c != "category"]
    if len(frame):
        frame[numeric] = frame[numeric].astype(np.float32)
    return frame
