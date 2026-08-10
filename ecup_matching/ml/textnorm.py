from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_MODEL_RE = re.compile(r"\b(?=[a-zа-я0-9_-]*[a-zа-я])(?=[a-zа-я0-9_-]*\d)[a-zа-я0-9_-]{2,}\b", re.I)
_QUANTITY_RE = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(kg|кг|g|гр|г|mg|мг|l|л|ml|мл|cm|см|mm|мм|m|м|pcs|pc|шт)\b",
    re.I,
)

_UNIT_MAP = {
    "kg": ("mass_g", 1000.0), "кг": ("mass_g", 1000.0),
    "g": ("mass_g", 1.0), "гр": ("mass_g", 1.0), "г": ("mass_g", 1.0),
    "mg": ("mass_g", 0.001), "мг": ("mass_g", 0.001),
    "l": ("volume_ml", 1000.0), "л": ("volume_ml", 1000.0),
    "ml": ("volume_ml", 1.0), "мл": ("volume_ml", 1.0),
    "m": ("length_mm", 1000.0), "м": ("length_mm", 1000.0),
    "cm": ("length_mm", 10.0), "см": ("length_mm", 10.0),
    "mm": ("length_mm", 1.0), "мм": ("length_mm", 1.0),
    "pcs": ("count", 1.0), "pc": ("count", 1.0), "шт": ("count", 1.0),
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    text = re.sub(r"[^\w.,%+\-/ ]+", " ", text, flags=re.UNICODE)
    return _WS_RE.sub(" ", text).strip()


def _flatten_json(value: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            k = clean_text(key)
            path = f"{prefix}.{k}" if prefix else k
            out.update(_flatten_json(sub, path))
    elif isinstance(value, list):
        vals = [clean_text(x) for x in value if clean_text(x)]
        if prefix and vals:
            out[prefix] = " | ".join(sorted(vals))
    else:
        if prefix:
            out[prefix] = clean_text(value)
    return out


def parse_attributes(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        obj = raw
    else:
        try:
            if isinstance(raw, float) and math.isnan(raw):
                return {}
        except TypeError:
            pass
        try:
            obj = json.loads(str(raw))
        except Exception:
            return {}
    return _flatten_json(obj) if isinstance(obj, (dict, list)) else {}


def tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN_RE.findall(text) if t)


def char_ngrams(text: str, n: int = 3) -> frozenset[str]:
    compact = text.replace(" ", "")
    if not compact:
        return frozenset()
    if len(compact) < n:
        return frozenset({compact})
    return frozenset(compact[i : i + n] for i in range(len(compact) - n + 1))


def extract_numbers(text: str) -> frozenset[str]:
    return frozenset(m.group(0).replace(",", ".") for m in _NUMBER_RE.finditer(text))


def extract_model_codes(text: str) -> frozenset[str]:
    result = set()
    for token in _MODEL_RE.findall(text):
        token = token.lower().strip("-_")
        if token and not re.fullmatch(r"\d+(?:gb|гб|mb|мб|tb|тб)", token):
            result.add(token)
    return frozenset(result)


def extract_quantities(text: str) -> frozenset[tuple[str, float]]:
    result: set[tuple[str, float]] = set()
    for number, unit in _QUANTITY_RE.findall(text):
        dim, multiplier = _UNIT_MAP[unit.lower()]
        value = round(float(number.replace(",", ".")) * multiplier, 6)
        result.add((dim, value))
    return frozenset(result)


@dataclass(frozen=True)
class ItemNorm:
    item_id: object
    category: str
    name: str
    attrs: dict[str, str]
    name_tokens: frozenset[str]
    name_char3: frozenset[str]
    attr_tokens: frozenset[str]
    numbers: frozenset[str]
    model_codes: frozenset[str]
    quantities: frozenset[tuple[str, float]]


def normalize_item(item_id: object, name: Any, attributes: Any, category: Any) -> ItemNorm:
    norm_name = clean_text(name)
    attrs = parse_attributes(attributes)
    attr_text = " ".join(f"{k} {v}" for k, v in sorted(attrs.items()))
    combined = f"{norm_name} {attr_text}".strip()
    return ItemNorm(
        item_id=item_id,
        category=clean_text(category),
        name=norm_name,
        attrs=attrs,
        name_tokens=tokens(norm_name),
        name_char3=char_ngrams(norm_name),
        attr_tokens=tokens(attr_text),
        numbers=extract_numbers(combined),
        model_codes=extract_model_codes(combined),
        quantities=extract_quantities(combined),
    )
