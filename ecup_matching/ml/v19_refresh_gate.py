from __future__ import annotations

from typing import Mapping

from .v18_metrics import worst_qualifying_category_delta

_EPS = 1e-12


def _validate(metrics: Mapping[str, object]) -> None:
    if metrics.get("gold_metric_opened") is not False:
        raise ValueError("sealed gold provenance violation")
    if int(metrics.get("cross_split_item_overlap", -1)) != 0:
        raise ValueError("cross-split item overlap must be zero")


def evaluate_refresh(pre: Mapping[str, object], post: Mapping[str, object]) -> dict[str, object]:
    _validate(pre)
    _validate(post)
    human_delta = float(post["human_macro_average_precision"]) - float(pre["human_macro_average_precision"])
    weak_delta = float(post["weak_macro_average_precision"]) - float(pre["weak_macro_average_precision"])
    brier_delta = float(post["weak_soft_brier"]) - float(pre["weak_soft_brier"])
    category = worst_qualifying_category_delta(
        candidate_per_category=post["per_category_ap"],
        control_per_category=pre["per_category_ap"],
        category_rows=post["category_row_counts"],
        min_rows=200,
    )
    weak_gate = weak_delta > 0.005 + _EPS
    human_gate = human_delta >= -0.002 - _EPS
    category_gate = float(category["worst_delta"]) >= -0.03 - _EPS
    brier_gate = brier_delta <= 0.002 + _EPS
    return {
        "human_delta": float(human_delta),
        "weak_delta": float(weak_delta),
        "weak_brier_delta": float(brier_delta),
        "category": category,
        "weak_gate": bool(weak_gate),
        "human_gate": bool(human_gate),
        "category_gate": bool(category_gate),
        "brier_gate": bool(brier_gate),
        "promote": bool(weak_gate and human_gate and category_gate and brier_gate),
        "rule": "refresh: weak>+0.005, human>=-0.002, worst_category>=-0.03, brier_delta<=+0.002",
    }


def evaluate_two_fold_refresh(
    pre0: Mapping[str, object],
    post0: Mapping[str, object],
    pre1: Mapping[str, object],
    post1: Mapping[str, object],
) -> dict[str, object]:
    fold0 = evaluate_refresh(pre0, post0)
    fold1 = evaluate_refresh(pre1, post1)
    mean_human = (float(fold0["human_delta"]) + float(fold1["human_delta"])) / 2.0
    mean_weak = (float(fold0["weak_delta"]) + float(fold1["weak_delta"])) / 2.0
    mean_human_gate = mean_human >= 0.0 - _EPS
    mean_weak_gate = mean_weak > 0.005 + _EPS
    return {
        "fold0": fold0,
        "fold1": fold1,
        "mean_human_delta": float(mean_human),
        "mean_weak_delta": float(mean_weak),
        "mean_human_gate": bool(mean_human_gate),
        "mean_weak_gate": bool(mean_weak_gate),
        "promote": bool(fold0["promote"] and fold1["promote"] and mean_human_gate and mean_weak_gate),
        "rule": "two-fold refresh: each fold passes; mean human>=0; mean weak>+0.005",
    }


__all__ = ["evaluate_refresh", "evaluate_two_fold_refresh"]
