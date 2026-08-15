"""Symmetric deterministic pair evidence for E-CUP v15."""

from __future__ import annotations

import re
import numpy as np

from .v15_fields import NormalizedItemFields

PAIR_FEATURE_NAMES = (
    "brand_equal",
    "brand_conflict",
    "model_exact",
    "model_overlap",
    "model_conflict",
    "numeric_overlap_count",
    "numeric_conflict_count",
    "title_token_jaccard",
    "attribute_key_jaccard",
    "attribute_value_agreement_count",
    "attribute_value_conflict_count",
    "attribute_parse_both_ok",
    "title_length_ratio",
)

_TOKEN_RE = re.compile(r"[\w.-]+", re.UNICODE)


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def _title_tokens(item: NormalizedItemFields) -> set[str]:
    return set(_TOKEN_RE.findall(item.title))


def _attr_map(item: NormalizedItemFields) -> dict[str, str]:
    return dict(item.attributes)


def build_pair_features(a: NormalizedItemFields, b: NormalizedItemFields) -> np.ndarray:
    brand_equal = float(bool(a.brand and b.brand and a.brand == b.brand))
    brand_conflict = float(bool(a.brand and b.brand and a.brand != b.brand))

    models_a, models_b = set(a.model_tokens), set(b.model_tokens)
    model_exact = float(bool(models_a and models_b and models_a == models_b))
    model_overlap = _jaccard(models_a, models_b)
    model_conflict = float(bool(models_a and models_b and not (models_a & models_b)))

    nums_a, nums_b = set(a.numeric_tokens), set(b.numeric_tokens)
    numeric_overlap_count = float(len(nums_a & nums_b))
    # A symmetric magnitude of disagreement. Identical/no-number pairs are 0.
    numeric_conflict_count = float(len(nums_a ^ nums_b))

    title_jaccard = _jaccard(_title_tokens(a), _title_tokens(b))

    attrs_a, attrs_b = _attr_map(a), _attr_map(b)
    keys_a, keys_b = set(attrs_a), set(attrs_b)
    attr_key_jaccard = _jaccard(keys_a, keys_b)
    common_keys = keys_a & keys_b
    attr_agree = float(sum(attrs_a[k] == attrs_b[k] for k in common_keys))
    attr_conflict = float(sum(attrs_a[k] != attrs_b[k] for k in common_keys))

    parse_both = float(a.raw_attributes_parse_ok and b.raw_attributes_parse_ok)
    la, lb = len(a.title), len(b.title)
    title_length_ratio = float(min(la, lb) / max(la, lb)) if max(la, lb) else 1.0

    return np.asarray(
        [
            brand_equal,
            brand_conflict,
            model_exact,
            model_overlap,
            model_conflict,
            numeric_overlap_count,
            numeric_conflict_count,
            title_jaccard,
            attr_key_jaccard,
            attr_agree,
            attr_conflict,
            parse_both,
            title_length_ratio,
        ],
        dtype=np.float32,
    )
