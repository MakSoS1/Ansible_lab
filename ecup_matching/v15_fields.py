"""Deterministic field normalization for the E-CUP v15 matcher.

This module deliberately contains no learned/global state and no network access.
It is shared by training and submission inference so field semantics cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Any, Iterable

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"(?<![\w.])\d+(?:[\.,]\d+)?(?![\w.])")
_ALNUM_RE = re.compile(r"(?=[\w-]*[a-zа-я])(?=[\w-]*\d)[a-zа-я0-9][a-zа-я0-9._/-]*", re.IGNORECASE)

_BRAND_KEYS = ("brand", "бренд", "марка", "производитель")
_MODEL_KEYS = ("model", "модель", "sku", "артикул", "part number", "part_number", "mpn", "код модели")


@dataclass(frozen=True)
class NormalizedItemFields:
    title: str
    category: str
    brand: str
    model_tokens: tuple[str, ...]
    numeric_tokens: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...]
    raw_attributes_parse_ok: bool


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower().replace("ё", "е")
    return _WS_RE.sub(" ", text)


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return normalize_text(str(value))
    if isinstance(value, int):
        return str(value)
    return normalize_text(value)


def _flatten(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    """Flatten JSON conservatively into deterministic key/value leaves.

    Nested keys keep their path so collisions are not silently merged. Lists of
    scalars are joined; lists of objects are indexed. Unsupported values are
    ignored rather than stringifying Python representations.
    """

    if isinstance(value, dict):
        for key in sorted(value, key=lambda x: normalize_text(x)):
            key_norm = normalize_text(key)
            child = f"{prefix}.{key_norm}" if prefix else key_norm
            yield from _flatten(value[key], child)
        return
    if isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            parts = [_scalar_text(v) for v in value]
            parts = [p for p in parts if p]
            if prefix and parts:
                yield prefix, " | ".join(parts)
        else:
            for idx, child in enumerate(value):
                yield from _flatten(child, f"{prefix}[{idx}]" if prefix else f"[{idx}]")
        return
    if prefix:
        scalar = _scalar_text(value)
        if scalar:
            yield prefix, scalar


def _leaf_key(path: str) -> str:
    leaf = path.rsplit(".", 1)[-1]
    return re.sub(r"\[\d+\]$", "", leaf)


def _extract_brand(attributes: tuple[tuple[str, str], ...]) -> str:
    for key, value in attributes:
        leaf = _leaf_key(key)
        if any(alias == leaf or alias in leaf for alias in _BRAND_KEYS):
            return value
    return ""


def _model_tokens_from_text(text: str) -> set[str]:
    out: set[str] = set()
    for match in _ALNUM_RE.findall(text):
        token = match.strip("._/-")
        if token and any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
            out.add(token)
    return out


def _extract_model_tokens(title: str, attributes: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    tokens = _model_tokens_from_text(title)
    for key, value in attributes:
        leaf = _leaf_key(key)
        if any(alias == leaf or alias in leaf for alias in _MODEL_KEYS):
            compact = re.sub(r"\s+", "", value)
            if compact:
                tokens.add(compact)
            tokens.update(_model_tokens_from_text(value))
    return tuple(sorted(tokens))


def _extract_numeric_tokens(*texts: str) -> tuple[str, ...]:
    tokens: set[str] = set()
    for text in texts:
        for match in _NUM_RE.findall(text):
            token = match.replace(",", ".")
            if token.endswith(".0"):
                token = token[:-2]
            tokens.add(token)
    return tuple(sorted(tokens, key=lambda s: (float(s), s)))


def normalize_item_fields(name: str | None, attributes: str | None, category: str | None) -> NormalizedItemFields:
    title = normalize_text(name)
    category_norm = normalize_text(category)

    parsed_ok = False
    flattened: tuple[tuple[str, str], ...] = ()
    raw = "" if attributes is None else str(attributes)
    if raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                flattened = tuple(sorted(_flatten(parsed), key=lambda kv: (kv[0], kv[1])))
                parsed_ok = True
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed_ok = False
    elif raw.strip() == "":
        # Empty attribute payload is safely handled but is not evidence of a
        # successfully parsed JSON object.
        parsed_ok = False

    brand = _extract_brand(flattened)
    model_tokens = _extract_model_tokens(title, flattened)
    attr_text = " ".join(f"{k} {v}" for k, v in flattened)
    numeric_tokens = _extract_numeric_tokens(title, attr_text)

    return NormalizedItemFields(
        title=title,
        category=category_norm,
        brand=brand,
        model_tokens=model_tokens,
        numeric_tokens=numeric_tokens,
        attributes=flattened,
        raw_attributes_parse_ok=parsed_ok,
    )
