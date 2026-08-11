from __future__ import annotations

from collections import defaultdict

from .textnorm import ItemNorm


def _quantities_by_dimension(item: ItemNorm) -> dict[str, frozenset[float]]:
    grouped: dict[str, set[float]] = defaultdict(set)
    for dimension, value in item.quantities:
        grouped[str(dimension)].add(float(value))
    return {dimension: frozenset(values) for dimension, values in grouped.items()}


def typed_quantity_consistency(left: ItemNorm, right: ItemNorm) -> float:
    """Return label-free typed quantity agreement in [-1, 1].

    Only dimensions present on both sides are comparable. Exact canonical value
    overlap contributes +1, a present-on-both conflict contributes -1, and
    dimensions missing on either side do not enter the denominator.
    """
    a = _quantities_by_dimension(left)
    b = _quantities_by_dimension(right)
    comparable = sorted(set(a) & set(b))
    if not comparable:
        return 0.0
    signed = 0
    for dimension in comparable:
        signed += 1 if a[dimension] & b[dimension] else -1
    return float(signed / len(comparable))
