from __future__ import annotations

import math

WEAK_IMPROVEMENT_FLOOR = 0.005
HUMAN_DROP_FLOOR = -0.005


def evaluate_promotion(
    *,
    control_weak: float,
    scaled_weak: float,
    control_human: float,
    scaled_human: float,
) -> dict[str, object]:
    values = [
        float(control_weak),
        float(scaled_weak),
        float(control_human),
        float(scaled_human),
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("metrics must be finite")

    weak_delta = values[1] - values[0]
    human_delta = values[3] - values[2]
    weak_gate = weak_delta > WEAK_IMPROVEMENT_FLOOR and not math.isclose(
        weak_delta,
        WEAK_IMPROVEMENT_FLOOR,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    human_gate = human_delta > HUMAN_DROP_FLOOR or math.isclose(
        human_delta,
        HUMAN_DROP_FLOOR,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    return {
        "control_weak": values[0],
        "scaled_weak": values[1],
        "weak_delta": weak_delta,
        "control_human": values[2],
        "scaled_human": values[3],
        "human_delta": human_delta,
        "weak_gate": bool(weak_gate),
        "human_gate": bool(human_gate),
        "promote": bool(weak_gate and human_gate),
        "weak_improvement_floor_exclusive": WEAK_IMPROVEMENT_FLOOR,
        "human_drop_floor_inclusive": HUMAN_DROP_FLOOR,
    }
