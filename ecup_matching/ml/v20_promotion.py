from __future__ import annotations

import math
from typing import Iterable, Mapping

from .v20_policy import V20Policy


def _strict_gt(value: float, threshold: float) -> bool:
    v, t = float(value), float(threshold)
    return bool(v > t and not math.isclose(v, t, rel_tol=0.0, abs_tol=1e-12))


def _ge(value: float, threshold: float) -> bool:
    v, t = float(value), float(threshold)
    return bool(v > t or math.isclose(v, t, rel_tol=0.0, abs_tol=1e-12))


def evaluate_candidate(
    *,
    proxy_delta: float,
    human_delta: float,
    audited_tail_delta: float,
    category_deltas: Mapping[str, float],
    proxy_axis_promotable: bool,
    policy: V20Policy | None = None,
) -> dict[str, object]:
    policy = policy or V20Policy()
    if not category_deltas:
        raise ValueError("category_deltas must not be empty")
    proxy_gate = bool(proxy_axis_promotable) and _strict_gt(proxy_delta, policy.proxy_gain_min_strict)
    human_gate = _ge(human_delta, policy.human_delta_min)
    audited_tail_gate = _ge(audited_tail_delta, policy.audited_tail_delta_min)
    worst_category = min(float(v) for v in category_deltas.values())
    category_gate = _ge(worst_category, policy.category_delta_min)
    promote = bool(proxy_gate and human_gate and audited_tail_gate and category_gate)
    return {
        "promote": promote,
        "proxy_gate": proxy_gate,
        "human_gate": human_gate,
        "audited_tail_gate": audited_tail_gate,
        "category_gate": category_gate,
        "proxy_axis_promotable": bool(proxy_axis_promotable),
        "proxy_delta": float(proxy_delta),
        "human_delta": float(human_delta),
        "audited_tail_delta": float(audited_tail_delta),
        "worst_category_delta": float(worst_category),
        "thresholds": {
            "proxy_delta_strict_gt": float(policy.proxy_gain_min_strict),
            "human_delta_ge": float(policy.human_delta_min),
            "audited_tail_delta_ge": float(policy.audited_tail_delta_min),
            "category_delta_ge": float(policy.category_delta_min),
        },
    }


def evaluate_scaled_confirmation(
    folds: Iterable[Mapping[str, object]],
    *,
    policy: V20Policy | None = None,
) -> dict[str, object]:
    policy = policy or V20Policy()
    rows = [dict(row) for row in folds]
    if len(rows) < 2:
        raise ValueError("scaled confirmation requires at least two folds")
    each_fold = all(bool(row.get("promote", False)) for row in rows)
    mean_human = sum(float(row["human_delta"]) for row in rows) / len(rows)
    mean_gate = _ge(mean_human, policy.scaled_mean_human_delta_min)
    return {
        "promote": bool(each_fold and mean_gate),
        "each_fold_gate": bool(each_fold),
        "mean_human_gate": bool(mean_gate),
        "mean_human_delta": float(mean_human),
        "folds": rows,
    }


__all__ = ["evaluate_candidate", "evaluate_scaled_confirmation"]
