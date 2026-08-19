from __future__ import annotations

import math
from typing import Mapping

_EPS = 1e-12


def _validate_metrics(metrics: Mapping[str, object]) -> None:
    if metrics.get("gold_metric_opened") is not False:
        raise ValueError("sealed gold provenance violation")
    if int(metrics.get("gold_rows_scored", -1)) != 0:
        raise ValueError("sealed gold rows must not be scored")
    if int(metrics.get("cross_split_item_overlap", -1)) != 0:
        raise ValueError("cross-split item overlap must be zero")


def _finite(metrics: Mapping[str, object], key: str) -> float:
    out = float(metrics[key])
    if not math.isfinite(out):
        raise ValueError(f"{key} must be finite")
    return out


def _worst_category_delta(control: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, object]:
    c = {str(k): float(v) for k, v in dict(control["per_category_ap"]).items()}
    a = {str(k): float(v) for k, v in dict(candidate["per_category_ap"]).items()}
    rows = {str(k): int(v) for k, v in dict(candidate["category_row_counts"]).items()}
    eligible = sorted(k for k, n in rows.items() if n >= 200 and k in c and k in a)
    if not eligible:
        raise ValueError("no qualifying category for transfer gate")
    deltas = {k: float(a[k] - c[k]) for k in eligible}
    worst = min(deltas, key=lambda k: (deltas[k], k))
    return {"worst_category": worst, "worst_delta": float(deltas[worst]), "deltas": deltas}


def _validate_calibration(calibration: Mapping[str, object]) -> None:
    if calibration.get("calibrated") is not True:
        raise ValueError("v21 Public proxy is not calibrated")
    if calibration.get("best_anchor") != "v19":
        raise ValueError("v19 must be the strongest empirical Public anchor")
    for key in (
        "normalized_gap_above_v19",
        "required_normalized_gap_for_target",
        "proxy_implied_public_lb",
        "target_public_lb",
    ):
        value = float(calibration[key])
        if not math.isfinite(value):
            raise ValueError(f"calibration {key} must be finite")


def evaluate_transfer_fold(
    control: Mapping[str, object],
    candidate: Mapping[str, object],
    calibration: Mapping[str, object],
) -> dict[str, object]:
    _validate_metrics(control)
    _validate_metrics(candidate)
    _validate_calibration(calibration)

    human_delta = _finite(candidate, "human_macro_average_precision") - _finite(control, "human_macro_average_precision")
    weak_delta = _finite(candidate, "weak_macro_average_precision") - _finite(control, "weak_macro_average_precision")
    brier_delta = _finite(candidate, "weak_soft_brier") - _finite(control, "weak_soft_brier")
    tail_delta = _finite(candidate, "tail_macro_average_precision") - _finite(control, "tail_macro_average_precision")
    category = _worst_category_delta(control, candidate)

    human_gate = human_delta >= -0.002 - _EPS
    weak_gate = weak_delta > 0.005 + _EPS
    brier_gate = brier_delta <= 0.002 + _EPS
    tail_gate = tail_delta >= -0.01 - _EPS
    category_gate = float(category["worst_delta"]) >= -0.03 - _EPS

    implied = float(calibration["proxy_implied_public_lb"])
    normalized = float(calibration["normalized_gap_above_v19"])
    required = float(calibration["required_normalized_gap_for_target"])
    target = float(calibration["target_public_lb"])
    public_target_gate = bool(
        calibration.get("candidate_above_v19") is True
        and calibration.get("target_reached") is True
        and normalized + _EPS >= required
        and implied + _EPS >= target
        and target >= 0.5 - _EPS
    )

    promote = bool(
        human_gate and weak_gate and brier_gate and tail_gate and category_gate and public_target_gate
    )
    return {
        "version": "v21-transfer-fold-gate-v1",
        "human_delta": float(human_delta),
        "weak_delta": float(weak_delta),
        "weak_brier_delta": float(brier_delta),
        "tail_delta": float(tail_delta),
        "category": category,
        "human_gate": bool(human_gate),
        "weak_gate": bool(weak_gate),
        "brier_gate": bool(brier_gate),
        "tail_gate": bool(tail_gate),
        "category_gate": bool(category_gate),
        "public_target_gate": bool(public_target_gate),
        "proxy_implied_public_lb": float(implied),
        "promote": bool(promote),
        "rule": "human>=-0.002; weak>+0.005; brier<=+0.002; tail>=-0.01; worst_category>=-0.03; proxy-implied Public>=0.5",
    }


def evaluate_two_fold_transfer(
    control0: Mapping[str, object],
    candidate0: Mapping[str, object],
    calibration0: Mapping[str, object],
    control1: Mapping[str, object],
    candidate1: Mapping[str, object],
    calibration1: Mapping[str, object],
) -> dict[str, object]:
    fold0 = evaluate_transfer_fold(control0, candidate0, calibration0)
    fold1 = evaluate_transfer_fold(control1, candidate1, calibration1)
    mean_human = (float(fold0["human_delta"]) + float(fold1["human_delta"])) / 2.0
    mean_weak = (float(fold0["weak_delta"]) + float(fold1["weak_delta"])) / 2.0
    min_implied = min(float(fold0["proxy_implied_public_lb"]), float(fold1["proxy_implied_public_lb"]))
    mean_human_gate = mean_human >= 0.0 - _EPS
    mean_weak_gate = mean_weak > 0.005 + _EPS
    public_target_gate = min_implied >= 0.5 - _EPS
    promote = bool(
        fold0["promote"] and fold1["promote"] and mean_human_gate and mean_weak_gate and public_target_gate
    )
    return {
        "version": "v21-two-fold-transfer-gate-v1",
        "fold0": fold0,
        "fold1": fold1,
        "mean_human_delta": float(mean_human),
        "mean_weak_delta": float(mean_weak),
        "min_proxy_implied_public_lb": float(min_implied),
        "mean_human_gate": bool(mean_human_gate),
        "mean_weak_gate": bool(mean_weak_gate),
        "public_target_gate": bool(public_target_gate),
        "promote": bool(promote),
        "rule": "both folds pass; mean human>=0; mean weak>+0.005; both proxy-implied Public>=0.5",
    }


__all__ = ["evaluate_transfer_fold", "evaluate_two_fold_transfer"]
