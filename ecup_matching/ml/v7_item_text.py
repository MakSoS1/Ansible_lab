from __future__ import annotations

from collections.abc import Mapping

from .textnorm import (
    ItemNorm,
    canonical_attribute_value,
    canonical_model_code,
    clean_text,
)


_IDENTITY_KEY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("памят", "storage", "накопител", "ssd", "hdd"),
    ("аккумуля", "battery", "mah", "мач"),
    ("диагон", "display", "экран", "screen", "inch", "дюйм"),
    ("мощност", "power", "watt", "ватт"),
    ("частот", "frequency", "hz", "гц"),
    ("напряж", "voltage", "volt", "вольт"),
    ("размер", "габарит", "длин", "ширин", "высот", "length", "width", "height"),
    ("вес", "масса", "weight"),
    ("объем", "обьем", "volume"),
    ("колич", "комплект", "count", "шт"),
    ("цвет", "color"),
    ("материал", "material", "состав", "fabric", "ткан"),
    ("пол", "gender", "sex", "для кого"),
    ("сезон", "season"),
    ("проба", "hallmark", "карат"),
    ("вставк", "камень", "stone", "gem"),
)

_TYPED_VALUE_PREFIXES: tuple[str, ...] = (
    "storage_bytes_",
    "battery_mah_",
    "diagonal_in_",
    "power_w_",
    "frequency_hz_",
    "voltage_v_",
    "length_mm_",
    "mass_g_",
    "volume_ml_",
    "count_",
)

_BRAND_KEYS = ("бренд", "brand", "марка", "manufacturer", "производитель")
_MODEL_KEYS = ("модель", "model", "sku", "артикул", "part number", "part_number", "mpn")


def _key_matches(key: str, needles: tuple[str, ...]) -> bool:
    key = clean_text(key)
    return any(needle in key for needle in needles)


def _first_attribute(item: ItemNorm, needles: tuple[str, ...]) -> str:
    for key, value in sorted(item.attrs.items()):
        if _key_matches(key, needles):
            normalized = canonical_attribute_value(value)
            if normalized:
                return normalized
    return ""


def _brand(item: ItemNorm) -> str:
    return _first_attribute(item, _BRAND_KEYS)


def _model_codes(item: ItemNorm) -> list[str]:
    explicit = _first_attribute(item, _MODEL_KEYS)
    result: list[str] = []
    if explicit:
        for token in explicit.replace("|", " ").split():
            candidate = canonical_model_code(token)
            if candidate and any(ch.isalpha() for ch in candidate) and any(ch.isdigit() for ch in candidate):
                result.append(candidate)
    for code in sorted(item.model_codes):
        candidate = canonical_model_code(code)
        if candidate and candidate not in result:
            result.append(candidate)
    return result


def _identity_priority(key: str, canonical_value: str) -> tuple[int, int, str]:
    for index, prefix in enumerate(_TYPED_VALUE_PREFIXES):
        if prefix in canonical_value:
            return (0, index, clean_text(key))
    normalized_key = clean_text(key)
    for index, needles in enumerate(_IDENTITY_KEY_GROUPS):
        if any(needle in normalized_key for needle in needles):
            return (1, index, normalized_key)
    return (2, len(_IDENTITY_KEY_GROUPS), normalized_key)


def _identity_attributes(item: ItemNorm) -> tuple[list[str], list[str]]:
    critical: list[tuple[tuple[int, int, str], str, str]] = []
    residual: list[tuple[str, str]] = []
    for raw_key, raw_value in sorted(item.attrs.items()):
        key = clean_text(raw_key)
        if not key or _key_matches(key, _BRAND_KEYS) or _key_matches(key, _MODEL_KEYS):
            continue
        value = canonical_attribute_value(raw_value)
        if not value:
            continue
        priority = _identity_priority(key, value)
        entry = f"{key}={value}"
        if priority[0] < 2:
            critical.append((priority, key, entry))
        else:
            residual.append((key, entry))
    critical.sort(key=lambda row: (row[0], row[1]))
    residual.sort(key=lambda row: row[0])
    return [entry for _, _, entry in critical], [entry for _, entry in residual]


def _append_line(lines: list[str], line: str, max_chars: int) -> bool:
    candidate = "\n".join([*lines, line]) if lines else line
    if len(candidate) > max_chars:
        return False
    lines.append(line)
    return True


def _append_entries(lines: list[str], label: str, entries: list[str], max_chars: int) -> None:
    if not entries:
        return
    accepted: list[str] = []
    for entry in entries:
        line = f"{label} " + " | ".join([*accepted, entry])
        candidate = "\n".join([*lines, line]) if lines else line
        if len(candidate) > max_chars:
            break
        accepted.append(entry)
    if accepted:
        lines.append(f"{label} " + " | ".join(accepted))


def serialize_item_v7(
    item: ItemNorm,
    *,
    max_chars: int = 900,
    attribute_importance: Mapping[str, float] | None = None,
) -> str:
    """Serialize an item for the v7 cross-encoder with identity evidence first.

    The retained v5 teacher placed a potentially long generic numeric section before
    attributes. v7 intentionally keeps name/brand/model and canonical identity
    attributes at the front, then spends remaining context on numeric/residual data.
    `attribute_importance` is optional and only affects the residual tie-break; the
    identity packet itself stays deterministic and schema-driven.
    """
    if max_chars < 64:
        raise ValueError("max_chars must be at least 64")

    lines: list[str] = []
    name_budget = min(140, max(1, max_chars - len("[NAME] ")))
    _append_line(lines, f"[NAME] {item.name[:name_budget]}", max_chars)

    brand = _brand(item)
    if brand:
        _append_line(lines, f"[BRAND] {brand}", max_chars)

    models = _model_codes(item)
    if models:
        _append_entries(lines, "[MODEL]", models, max_chars)

    identity, residual = _identity_attributes(item)
    _append_entries(lines, "[IDENTITY]", identity, max_chars)

    represented = " ".join(identity)
    numeric: list[str] = []
    for dimension, value in sorted(item.quantities, key=lambda row: (row[0], row[1])):
        token = f"{dimension}_{int(value) if float(value).is_integer() else value:g}"
        if token not in represented:
            numeric.append(token)
    for number in sorted(item.numbers):
        if number not in represented:
            numeric.append(number)
    _append_entries(lines, "[NUMERIC]", numeric, max_chars)

    if attribute_importance:
        residual.sort(
            key=lambda entry: (
                -float(attribute_importance.get(entry.split("=", 1)[0], 0.0)),
                entry,
            )
        )
    _append_entries(lines, "[RESIDUAL]", residual, max_chars)

    text = "\n".join(lines)
    if len(text) > max_chars:
        raise RuntimeError("v7 serializer exceeded hard character bound")
    return text
