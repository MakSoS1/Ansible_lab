from __future__ import annotations

from .category_attrs import brand_from_item
from .textnorm import ItemNorm, clean_text


def _leaf(key: str) -> str:
    return clean_text(key).rsplit(".", 1)[-1]


def _append_with_budget(parts: list[str], section: str, max_chars: int) -> None:
    if not section:
        return
    prefix = "\n" if parts else ""
    used = sum(len(part) for part in parts) + max(0, len(parts) - 1)
    remaining = max_chars - used - len(prefix)
    if remaining <= 0:
        return
    parts.append(section[:remaining].rstrip())


def serialize_item_v5(item: ItemNorm, *, max_chars: int = 1200) -> str:
    """Serialize one normalized product with identity-critical fields first."""
    if max_chars < 64:
        raise ValueError("max_chars must be at least 64")

    parts: list[str] = []
    _append_with_budget(parts, f"[NAME] {item.name}", max_chars)

    brand = brand_from_item(item)
    if brand:
        _append_with_budget(parts, f"[BRAND] {brand}", max_chars)

    if item.model_codes:
        _append_with_budget(parts, "[MODEL] " + " ; ".join(sorted(item.model_codes)), max_chars)

    numeric_tokens = list(sorted(item.numbers))
    numeric_tokens.extend(
        f"{dimension}={value:g}" for dimension, value in sorted(item.quantities)
    )
    if numeric_tokens:
        _append_with_budget(parts, "[NUMERIC] " + " ; ".join(numeric_tokens), max_chars)

    # Deterministic attribute order. Brand/model-like facts have already been
    # surfaced in higher-priority sections but retaining them here is harmless
    # when budget remains and preserves the original key/value context.
    attr_tokens = [f"{_leaf(key)}={value}" for key, value in sorted(item.attrs.items()) if value]
    if attr_tokens:
        _append_with_budget(parts, "[ATTR] " + " ; ".join(attr_tokens), max_chars)

    text = "\n".join(parts)
    return text[:max_chars].rstrip()
