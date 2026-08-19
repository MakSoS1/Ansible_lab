from __future__ import annotations

from .textnorm import ItemNorm


def model_code_consistency(left: ItemNorm, right: ItemNorm) -> float:
    """Return deterministic model/SKU agreement in {-1, 0, 1}."""
    a = left.model_codes
    b = right.model_codes
    if not a or not b:
        return 0.0
    return 1.0 if a & b else -1.0
