from __future__ import annotations

import math
from typing import Mapping


def worst_qualifying_category_delta(
    *,
    candidate_per_category: Mapping[str, float],
    control_per_category: Mapping[str, float],
    category_rows: Mapping[str, int],
    min_rows: int = 200,
) -> dict[str, object]:
    deltas: dict[str, float] = {}
    for category, rows in category_rows.items():
        if int(rows) < int(min_rows):
            continue
        if category not in candidate_per_category or category not in control_per_category:
            continue
        candidate = float(candidate_per_category[category])
        control = float(control_per_category[category])
        if not math.isfinite(candidate) or not math.isfinite(control):
            continue
        deltas[str(category)] = candidate - control
    if not deltas:
        return {
            "qualifying_categories": 0,
            "worst_category": None,
            "worst_delta": 0.0,
            "per_category_delta": {},
            "min_rows": int(min_rows),
        }
    worst_category = min(deltas, key=lambda key: (deltas[key], key))
    return {
        "qualifying_categories": int(len(deltas)),
        "worst_category": worst_category,
        "worst_delta": float(deltas[worst_category]),
        "per_category_delta": {key: float(value) for key, value in sorted(deltas.items())},
        "min_rows": int(min_rows),
    }


def robust_gain_score(*, human_delta: float, weak_delta: float, worst_category_delta: float) -> float:
    values = [float(human_delta), float(weak_delta), float(worst_category_delta)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("robust gain inputs must be finite")
    # Human truth and broad weak-population agreement get equal primary weight.
    # Only negative tail movement is penalized; a single category cannot create
    # artificial upside that overwhelms the two aggregate axes.
    return float(0.45 * values[0] + 0.45 * values[1] + 0.10 * min(0.0, values[2]))


__all__ = ["robust_gain_score", "worst_qualifying_category_delta"]
