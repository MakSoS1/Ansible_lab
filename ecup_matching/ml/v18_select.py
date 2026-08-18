from __future__ import annotations

import math
from typing import Mapping

from .v18_metrics import robust_gain_score, worst_qualifying_category_delta

_EPS = 1e-12


def _validate(metrics: Mapping[str, object]) -> None:
    if metrics.get("gold_metric_opened") is not False:
        raise ValueError("sealed gold provenance violation")
    if int(metrics.get("cross_split_item_overlap", -1)) != 0:
        raise ValueError("cross-split item overlap must be zero")


def _summary(control: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, object]:
    _validate(control)
    _validate(candidate)
    human_delta = float(candidate["fold_macro_average_precision"]) - float(control["fold_macro_average_precision"])
    weak_delta = float(candidate["weak_holdout_after_human_phase"]["macro_average_precision"]) - float(
        control["weak_holdout_after_human_phase"]["macro_average_precision"]
    )
    category = worst_qualifying_category_delta(
        candidate_per_category=candidate["per_category_ap"],
        control_per_category=control["per_category_ap"],
        category_rows=candidate["category_row_counts"],
        min_rows=200,
    )
    robust = robust_gain_score(
        human_delta=human_delta,
        weak_delta=weak_delta,
        worst_category_delta=float(category["worst_delta"]),
    )
    return {
        "human_delta": float(human_delta),
        "weak_delta": float(weak_delta),
        "category": category,
        "robust_gain_score": float(robust),
    }


def evaluate_single(control: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, object]:
    result = _summary(control, candidate)
    weak_gate = float(result["weak_delta"]) > 0.003 + _EPS
    human_gate = float(result["human_delta"]) >= -0.003 - _EPS
    category_gate = float(result["category"]["worst_delta"]) >= -0.03 - _EPS
    result.update(
        weak_gate=bool(weak_gate),
        human_gate=bool(human_gate),
        category_gate=bool(category_gate),
        promote=bool(weak_gate and human_gate and category_gate),
        rule="single: weak>+0.003, human>=-0.003, worst_category>=-0.03",
    )
    return result


def select_mechanisms(
    control: Mapping[str, object],
    candidates: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    evaluations = {str(name): evaluate_single(control, metrics) for name, metrics in candidates.items()}
    selected = sorted(name for name, result in evaluations.items() if result["promote"])
    return {"selected": selected, "evaluations": evaluations, "no_keeper": not bool(selected)}


def evaluate_combination(
    control: Mapping[str, object],
    singles: Mapping[str, Mapping[str, object]],
    combined: Mapping[str, object],
) -> dict[str, object]:
    result = _summary(control, combined)
    single_results = {name: evaluate_single(control, metrics) for name, metrics in singles.items()}
    passing_scores = [
        float(value["robust_gain_score"])
        for value in single_results.values()
        if value["promote"]
    ]
    best_single = max(passing_scores) if passing_scores else -math.inf
    weak_gate = float(result["weak_delta"]) > 0.005 + _EPS
    human_gate = float(result["human_delta"]) >= -0.002 - _EPS
    category_gate = float(result["category"]["worst_delta"]) >= -0.03 - _EPS
    robust_gate = float(result["robust_gain_score"]) > best_single + _EPS
    result.update(
        weak_gate=bool(weak_gate),
        human_gate=bool(human_gate),
        category_gate=bool(category_gate),
        robust_gate=bool(robust_gate),
        best_single_robust_gain=float(best_single) if math.isfinite(best_single) else None,
        promote=bool(weak_gate and human_gate and category_gate and robust_gate),
        rule="combined: weak>+0.005, human>=-0.002, worst_category>=-0.03, robust>best_single",
    )
    return result


def evaluate_scaled_confirmation(
    fold0_control: Mapping[str, object],
    fold0_candidate: Mapping[str, object],
    fold1_control: Mapping[str, object],
    fold1_candidate: Mapping[str, object],
) -> dict[str, object]:
    s0 = _summary(fold0_control, fold0_candidate)
    s1 = _summary(fold1_control, fold1_candidate)
    h0 = float(s0["human_delta"])
    h1 = float(s1["human_delta"])
    w0 = float(s0["weak_delta"])
    w1 = float(s1["weak_delta"])
    c0 = float(s0["category"]["worst_delta"])
    c1 = float(s1["category"]["worst_delta"])
    human_fold_gate = h0 >= -0.002 - _EPS and h1 >= -0.002 - _EPS
    human_mean_gate = (h0 + h1) / 2.0 >= 0.0 - _EPS
    # Same item-disjoint weak split policy is used on both folds. Requiring the
    # weaker of the two post-human weak deltas to clear the threshold prevents
    # one human fine-tune from hiding forgetting in the other.
    weak_gate = min(w0, w1) > 0.005 + _EPS
    category_gate = min(c0, c1) >= -0.03 - _EPS
    return {
        "fold0": s0,
        "fold1": s1,
        "human_mean_delta": float((h0 + h1) / 2.0),
        "minimum_weak_delta": float(min(w0, w1)),
        "minimum_worst_category_delta": float(min(c0, c1)),
        "human_fold_gate": bool(human_fold_gate),
        "human_mean_gate": bool(human_mean_gate),
        "weak_gate": bool(weak_gate),
        "category_gate": bool(category_gate),
        "promote": bool(human_fold_gate and human_mean_gate and weak_gate and category_gate),
        "rule": "scaled: each human>=-0.002, mean human>=0, min weak>+0.005, worst_category>=-0.03",
    }


__all__ = [
    "evaluate_combination",
    "evaluate_scaled_confirmation",
    "evaluate_single",
    "select_mechanisms",
]
