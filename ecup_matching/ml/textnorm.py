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

# Keep the parser intentionally conservative: every short unit below must be
# sufficiently unambiguous when it appears immediately after a number. In
# particular we do not treat bare Russian "в" as volts because phrases such
# as "2 в 1" are common product text.
_UNIT_MAP = {
    # mass
    "kg": ("mass_g", 1000.0), "кг": ("mass_g", 1000.0),
    "g": ("mass_g", 1.0), "гр": ("mass_g", 1.0), "г": ("mass_g", 1.0),
    "mg": ("mass_g", 0.001), "мг": ("mass_g", 0.001),
    # volume
    "l": ("volume_ml", 1000.0), "л": ("volume_ml", 1000.0),
    "ml": ("volume_ml", 1.0), "мл": ("volume_ml", 1.0),
    # length
    "m": ("length_mm", 1000.0), "м": ("length_mm", 1000.0),
    "cm": ("length_mm", 10.0), "см": ("length_mm", 10.0),
    "mm": ("length_mm", 1.0), "мм": ("length_mm", 1.0),
    # count
    "pcs": ("count", 1.0), "pc": ("count", 1.0), "шт": ("count", 1.0),
    # storage, decimal vendor units
    "mb": ("storage_bytes", 1_000_000.0), "мб": ("storage_bytes", 1_000_000.0),
    "gb": ("storage_bytes", 1_000_000_000.0), "гб": ("storage_bytes", 1_000_000_000.0),
    "tb": ("storage_bytes", 1_000_000_000_000.0), "тб": ("storage_bytes", 1_000_000_000_000.0),
    # battery capacity
    "mah": ("battery_mah", 1.0), "мач": ("battery_mah", 1.0),
    # electrical power
    "w": ("power_w", 1.0), "вт": ("power_w", 1.0),
    "kw": ("power_w", 1000.0), "квт": ("power_w", 1000.0),
    "watt": ("power_w", 1.0), "watts": ("power_w", 1.0),
    "ватт": ("power_w", 1.0), "ватта": ("power_w", 1.0), "ваттa": ("power_w", 1.0),
    # voltage: deliberately no bare Cyrillic "в"
    "v": ("voltage_v", 1.0), "volt": ("voltage_v", 1.0), "volts": ("voltage_v", 1.0),
    "вольт": ("voltage_v", 1.0), "вольта": ("voltage_v", 1.0), "вольтa": ("voltage_v", 1.0),
    # frequency
    "hz": ("frequency_hz", 1.0), "гц": ("frequency_hz", 1.0),
    "khz": ("frequency_hz", 1_000.0), "кгц": ("frequency_hz", 1_000.0),
    "mhz": ("frequency_hz", 1_000_000.0), "мгц": ("frequency_hz", 1_000_000.0),
    "ghz": ("frequency_hz", 1_000_000_000.0), "ггц": ("frequency_hz", 1_000_000_000.0),
    # display diagonal; omit bare "in" to avoid "2 in 1" ambiguity
    "inch": ("diagonal_in", 1.0), "inches": ("diagonal_in", 1.0),
    "дюйм": ("diagonal_in", 1.0), "дюйма": ("diagonal_in", 1.0), "дюймов": ("diagonal_in", 1.0),
}
_UNIT_PATTERN = "|".join(sorted((re.escape(unit) for unit in _UNIT_MAP), key=len, reverse=True))
_QUANTITY_RE = re.compile(
    rf"(?<!\w)(\d+(?:[.,]\d+)?)\s*({_UNIT_PATTERN})(?!\w)",
    re.I,
)


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


def _canonical_quantity(number: str, unit: str) -> tuple[str, float]:
    dim, multiplier = _UNIT_MAP[unit.lower()]
    value = round(float(number.replace(",", ".")) * multiplier, 6)
    return dim, value


def _format_canonical_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def canonical_attribute_value(value: Any) -> str:
    """Normalize an attribute value while preserving typed quantity semantics.

    This is intentionally label-free. Equivalent unit spellings receive the
    same canonical marker while surrounding text remains part of the value.
    """
    text = clean_text(value)
    if not text:
        return ""

    def repl(match: re.Match[str]) -> str:
        dim, canonical = _canonical_quantity(match.group(1), match.group(2))
        return f" {dim}_{_format_canonical_number(canonical)} "

    return _WS_RE.sub(" ", _QUANTITY_RE.sub(repl, text)).strip()


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
    for match in _QUANTITY_RE.finditer(text):
        result.add(_canonical_quantity(match.group(1), match.group(2)))
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
