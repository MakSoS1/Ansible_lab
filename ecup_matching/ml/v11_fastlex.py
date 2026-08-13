from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .textnorm import ItemNorm, normalize_item


FEATURE_NAMES: tuple[str, ...] = (
    "category",
    "same_category",
    "name_exact",
    "name_contains",
    "name_token_jaccard",
    "name_token_containment",
    "name_char3_jaccard",
    "first_token_equal",
    "last_token_equal",
    "name_len_ratio",
    "name_token_count_ratio",
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


def _containment(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / min(len(sa), len(sb)))


def _set_conflict(a: frozenset, b: frozenset) -> float:
    return float(bool(a) and bool(b) and not bool(a & b))


def _quantity_conflict(
    a: frozenset[tuple[str, float]], b: frozenset[tuple[str, float]]
) -> float:
    if not a or not b:
        return 0.0
    by_dim_a: dict[str, set[float]] = {}
    by_dim_b: dict[str, set[float]] = {}
    for dim, value in a:
        by_dim_a.setdefault(dim, set()).add(value)
    for dim, value in b:
        by_dim_b.setdefault(dim, set()).add(value)
    for dim in set(by_dim_a) & set(by_dim_b):
        if by_dim_a[dim].isdisjoint(by_dim_b[dim]):
            return 1.0
    return 0.0


def _attr_stats(a: ItemNorm, b: ItemNorm) -> tuple[float, float, float, float, float]:
    keys_a, keys_b = set(a.attrs), set(b.attrs)
    shared = keys_a & keys_b
    key_jaccard = _jaccard(keys_a, keys_b)
    shared_ratio = (
        float(len(shared) / max(1, min(len(keys_a), len(keys_b))))
        if keys_a and keys_b
        else 0.0
    )
    if shared:
        equal = sum(a.attrs[k] == b.attrs[k] for k in shared)
        agreement = float(equal / len(shared))
        conflict = float(1.0 - agreement)
    else:
        agreement = 0.0
        conflict = 0.0
    return key_jaccard, shared_ratio, agreement, conflict, _jaccard(a.attr_tokens, b.attr_tokens)


def _edge_token_equal(a: frozenset[str], b: frozenset[str], first: bool) -> float:
    # ItemNorm stores token sets, so use sorted lexical edge as a cheap stable
    # proxy. It is deliberately deterministic and avoids edit-distance work.
    if not a or not b:
        return 0.0
    sa, sb = sorted(a), sorted(b)
    return float((sa[0] if first else sa[-1]) == (sb[0] if first else sb[-1]))


def _pair_features(a: ItemNorm, b: ItemNorm) -> tuple[object, ...]:
    name_a, name_b = a.name, b.name
    max_len = max(len(name_a), len(name_b), 1)
    min_len = min(len(name_a), len(name_b))
    max_tok = max(len(a.name_tokens), len(b.name_tokens), 1)
    min_tok = min(len(a.name_tokens), len(b.name_tokens))
    attr_key_j, attr_shared, attr_agree, attr_conflict, attr_token_j = _attr_stats(a, b)
    return (
        a.category or b.category or "__missing__",
        float(a.category == b.category),
        float(bool(name_a) and name_a == name_b),
        float(bool(name_a) and bool(name_b) and (name_a in name_b or name_b in name_a)),
        _jaccard(a.name_tokens, b.name_tokens),
        _containment(a.name_tokens, b.name_tokens),
        _jaccard(a.name_char3, b.name_char3),
        _edge_token_equal(a.name_tokens, b.name_tokens, True),
        _edge_token_equal(a.name_tokens, b.name_tokens, False),
        float(min_len / max_len),
        float(min_tok / max_tok),
        _jaccard(a.numbers, b.numbers),
        _set_conflict(a.numbers, b.numbers),
        _jaccard(a.model_codes, b.model_codes),
        _set_conflict(a.model_codes, b.model_codes),
        _jaccard(a.quantities, b.quantities),
        _quantity_conflict(a.quantities, b.quantities),
        attr_key_j,
        attr_shared,
        attr_agree,
        attr_conflict,
        attr_token_j,
        float(not name_a or not name_b),
        float(not a.attrs or not b.attrs),
    )


def build_item_cache(items: pd.DataFrame) -> dict[object, ItemNorm]:
    required = {"id", "name", "attributes", "category"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items is missing required columns: {sorted(missing)}")
    cache: dict[object, ItemNorm] = {}
    for item_id, name, attrs, category in items[
        ["id", "name", "attributes", "category"]
    ].itertuples(index=False, name=None):
        cache[item_id] = normalize_item(item_id, name, attrs, category)
    return cache


def build_fast_pair_features(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    cache: dict[object, ItemNorm] | None = None,
) -> pd.DataFrame:
    if not {"id1", "id2"}.issubset(pairs.columns):
        raise ValueError("pairs must contain id1 and id2")
    item_cache = cache if cache is not None else build_item_cache(items)
    rows: list[tuple[object, ...]] = []
    append = rows.append
    for id1, id2 in pairs[["id1", "id2"]].itertuples(index=False, name=None):
        try:
            a, b = item_cache[id1], item_cache[id2]
        except KeyError as exc:
            raise KeyError(f"pair references missing item: {exc.args[0]!r}") from exc
        append(_pair_features(a, b))
    frame = pd.DataFrame.from_records(rows, columns=FEATURE_NAMES)
    numeric = [name for name in FEATURE_NAMES if name != "category"]
    if len(frame):
        frame[numeric] = frame[numeric].astype(np.float32)
    return frame


__all__ = ["FEATURE_NAMES", "build_item_cache", "build_fast_pair_features"]
