"""Immutable v15 ablation definitions shared by GPU executor and documentation."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class V15Variant:
    name: str
    include_attributes: bool
    use_typed_features: bool
    use_category_head: bool
    macro_balanced: bool
    max_length: int = 128

_VARIANTS = {
    "A0": V15Variant("A0", False, False, False, False),
    "A1": V15Variant("A1", True, False, False, False),
    "A2": V15Variant("A2", True, True, False, False),
    "A3": V15Variant("A3", True, True, True, False),
    "A4": V15Variant("A4", True, True, True, True),
}

def get_variant(name: str) -> V15Variant:
    key = str(name).strip().upper()
    if key not in _VARIANTS:
        raise ValueError(f"unknown v15 variant: {name!r}; expected one of {sorted(_VARIANTS)}")
    return _VARIANTS[key]
