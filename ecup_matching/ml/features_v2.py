from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .category_attrs import weighted_attribute_stats
from .features import FEATURE_NAMES, _pair_features, _partial_ratio, _ratio, normalize_items
from .textnorm import ItemNorm


EXTRA_FEATURE_NAMES_V2: tuple[str, ...] = (
    "brand_match",
    "brand_conflict",
    "weighted_attr_agreement",
    "weighted_attr_conflict",
    "critical_conflict_count",
    "hard_negative_score",
)
FEATURE_NAMES_V2: tuple[str, ...] = (*FEATURE_NAMES, *EXTRA_FEATURE_NAMES_V2)


def _symmetric_base_features(a: ItemNorm, b: ItemNorm) -> dict[str, object]:
    """Match the original two-pass v2 symmetry with one full feature pass.

    All legacy v1 features are symmetric except the four SequenceMatcher-based
    fuzzy ratios. Compute the expensive complete vector once and average only
    those four values with their reverse-direction evaluations.
    """
    base = _pair_features(a, b)
    result = dict(base)

    category_a = str(a.category or b.category or "__missing__")
    category_b = str(b.category or a.category or "__missing__")
    result["category"] = category_a if category_a == category_b else min(category_a, category_b)

    sorted_a = " ".join(sorted(a.name_tokens))
    sorted_b = " ".join(sorted(b.name_tokens))
    common_tokens = sorted(a.name_tokens & b.name_tokens)
    only_a = sorted(a.name_tokens - b.name_tokens)
    only_b = sorted(b.name_tokens - a.name_tokens)
    token_set_a = " ".join(common_tokens + only_a)
    token_set_b = " ".join(common_tokens + only_b)

    reverse_values = {
        "fuzz_ratio": _ratio(b.name, a.name),
        "fuzz_partial_ratio": _partial_ratio(b.name, a.name),
        "fuzz_token_sort": _ratio(sorted_b, sorted_a),
        "fuzz_token_set": _ratio(token_set_b, token_set_a),
    }
    for name, reverse in reverse_values.items():
        result[name] = (float(base[name]) + float(reverse)) / 2.0
    return result


def _extra_features(
    a: ItemNorm,
    b: ItemNorm,
    base: Mapping[str, object],
    attribute_importance: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, float]:
    weighted = weighted_attribute_stats(a, b, attribute_importance)
    weighted_conflict = float(weighted["weighted_attr_conflict"])
    critical_conflicts = (
        float(base["number_conflict"])
        + float(base["model_code_conflict"])
        + float(base["quantity_conflict"])
        + float(weighted["brand_conflict"])
        + float(weighted_conflict >= 0.35)
    )

    lexical_similarity = (
        float(base["fuzz_token_set"])
        + float(base["name_token_jaccard"])
        + float(base["name_char3_jaccard"])
    ) / 3.0
    hard_negative_score = lexical_similarity * min(critical_conflicts, 5.0) / 5.0
    return {
        **weighted,
        "critical_conflict_count": float(critical_conflicts),
        "hard_negative_score": float(hard_negative_score),
    }


def build_pair_features_v2(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    attribute_importance: Mapping[str, Mapping[str, float]] | None = None,
    item_cache: dict[object, ItemNorm] | None = None,
) -> pd.DataFrame:
    if not {"id1", "id2"}.issubset(pairs.columns):
        raise ValueError("pairs must contain id1 and id2")
    cache = item_cache if item_cache is not None else normalize_items(items)
    rows: list[dict[str, object]] = []
    for id1, id2 in pairs[["id1", "id2"]].itertuples(index=False, name=None):
        if id1 not in cache or id2 not in cache:
            raise KeyError(f"pair references missing item: {id1!r}, {id2!r}")
        a, b = cache[id1], cache[id2]
        base = _symmetric_base_features(a, b)
        rows.append({**base, **_extra_features(a, b, base, attribute_importance)})

    frame = pd.DataFrame(rows, columns=FEATURE_NAMES_V2)
    numeric = [c for c in FEATURE_NAMES_V2 if c != "category"]
    if len(frame):
        frame[numeric] = frame[numeric].astype(np.float32)
    return frame


def build_features_v2_chunked(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    attribute_importance: Mapping[str, Mapping[str, float]] | None = None,
    chunk_size: int = 25_000,
) -> pd.DataFrame:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(pairs) == 0:
        return pd.DataFrame(columns=FEATURE_NAMES_V2)

    item_index = items.set_index("id", drop=False)
    chunks: list[pd.DataFrame] = []
    for start in range(0, len(pairs), chunk_size):
        pair_chunk = pairs.iloc[start : start + chunk_size]
        ids = pd.unique(pd.concat([pair_chunk["id1"], pair_chunk["id2"]], ignore_index=True))
        missing = [item_id for item_id in ids if item_id not in item_index.index]
        if missing:
            raise KeyError(f"pair chunk references {len(missing)} missing items; first={missing[0]!r}")
        item_chunk = item_index.loc[ids].reset_index(drop=True)
        chunks.append(
            build_pair_features_v2(
                item_chunk,
                pair_chunk,
                attribute_importance=attribute_importance,
            )
        )
    return pd.concat(chunks, ignore_index=True)
