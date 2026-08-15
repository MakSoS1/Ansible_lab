"""Stable field-aware text serialization for E-CUP v15."""

from __future__ import annotations

from .v15_fields import NormalizedItemFields


def serialize_item(item: NormalizedItemFields) -> str:
    parts: list[str] = [f"[TITLE] {item.title}"]
    if item.category:
        parts.append(f"[CATEGORY] {item.category}")
    if item.brand:
        parts.append(f"[BRAND] {item.brand}")
    if item.model_tokens:
        parts.append("[MODEL] " + " | ".join(item.model_tokens))
    if item.attributes:
        parts.append("[ATTR] " + " ; ".join(f"{key}={value}" for key, value in item.attributes))
    return " ".join(parts)


def serialize_pair(a: NormalizedItemFields, b: NormalizedItemFields) -> tuple[str, str]:
    """Return text/text_pair inputs for a pair tokenizer.

    Pair order is preserved here. Symmetry is enforced/evaluated at the model
    contract rather than by destroying endpoint identity in preprocessing.
    """

    return serialize_item(a), serialize_item(b)
