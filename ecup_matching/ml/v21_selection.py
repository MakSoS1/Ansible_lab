from __future__ import annotations

from typing import Mapping

from .v21_public_calibration import calibrate_candidate_public_lb

_EPS = 1e-12


def v20_metrics_to_transfer(metrics: Mapping[str, object]) -> dict[str, object]:
    per_category = {str(k): float(v) for k, v in dict(metrics["human_per_category_ap"]).items()}
    held_rows = int(metrics.get("held_rows", 0))
    supplied_counts = metrics.get("category_row_counts")
    if supplied_counts is None:
        # v20 probe did not persist per-category row counts.  For selection we make
        # every reported category eligible rather than silently dropping a tail.
        counts = {k: held_rows for k in per_category}
    else:
        counts = {str(k): int(v) for k, v in dict(supplied_counts).items()}
    weak = dict(metrics["weak_metrics"])
    return {
        "human_macro_average_precision": float(metrics["human_macro_average_precision"]),
        "weak_macro_average_precision": float(weak["macro_average_precision"]),
        "weak_soft_brier": float(weak["soft_brier"]),
        "tail_macro_average_precision": float(metrics["human_tail_macro_average_precision"]),
        "per_category_ap": per_category,
        "category_row_counts": counts,
        "gold_metric_opened": metrics.get("gold_metric_opened"),
        "gold_rows_scored": int(metrics.get("gold_rows_scored", -1)),
        "cross_split_item_overlap": int(metrics.get("cross_split_item_overlap", -1)),
    }


def _validate(m: Mapping[str, object]) -> None:
    if m.get("gold_metric_opened") is not False or int(m.get("gold_rows_scored", -1)) != 0:
        raise ValueError("sealed gold provenance violation")
    if int(m.get("cross_split_item_overlap", -1)) != 0:
        raise ValueError("cross-split item overlap must be zero")


def _safety(control: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, object]:
    _validate(control); _validate(candidate)
    human = float(candidate["human_macro_average_precision"]) - float(control["human_macro_average_precision"])
    weak = float(candidate["weak_macro_average_precision"]) - float(control["weak_macro_average_precision"])
    brier = float(candidate["weak_soft_brier"]) - float(control["weak_soft_brier"])
    tail = float(candidate["tail_macro_average_precision"]) - float(control["tail_macro_average_precision"])
    c0 = {str(k): float(v) for k, v in dict(control["per_category_ap"]).items()}
    c1 = {str(k): float(v) for k, v in dict(candidate["per_category_ap"]).items()}
    rows = {str(k): int(v) for k, v in dict(candidate["category_row_counts"]).items()}
    keys = [k for k in c0 if k in c1 and rows.get(k, 0) >= 200]
    if not keys:
        raise ValueError("no qualifying category for provisional selector")
    deltas = {k: c1[k] - c0[k] for k in keys}
    worst = min(deltas.values())
    gates = {
        "human_gate": human >= -0.002 - _EPS,
        "weak_gate": weak > 0.005 + _EPS,
        "brier_gate": brier <= 0.002 + _EPS,
        "tail_gate": tail >= -0.01 - _EPS,
        "category_gate": worst >= -0.03 - _EPS,
    }
    return {
        "human_delta": human, "weak_delta": weak, "weak_brier_delta": brier,
        "tail_delta": tail, "worst_category_delta": worst, **gates,
        "safe": bool(all(gates.values())),
    }


def select_provisional_keeper(
    control_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, Mapping[str, object]],
    anchor_proxy: Mapping[str, object],
) -> dict[str, object]:
    control = v20_metrics_to_transfer(control_metrics)
    evaluations: dict[str, object] = {}
    passing: list[tuple[float, float, float, str]] = []
    for name, raw in sorted(candidate_metrics.items()):
        candidate = v20_metrics_to_transfer(raw)
        safety = _safety(control, candidate)
        proxy_value = float(dict(raw["proxy_metrics"])["macro_average_precision"])
        calibration = calibrate_candidate_public_lb(anchor_proxy, candidate_proxy=proxy_value)
        evaluations[str(name)] = {**safety, "calibration": calibration}
        if safety["safe"]:
            passing.append((
                float(calibration["proxy_implied_public_lb"]),
                float(safety["weak_delta"]), float(safety["human_delta"]), str(name),
            ))
    selected = max(passing)[3] if passing else None
    return {
        "version": "v21-provisional-selection-v1",
        "selected": selected,
        "no_keeper": selected is None,
        "evaluations": evaluations,
        "rule": "fail-closed safety first; then maximize v19-calibrated proxy-implied Public LB",
    }


__all__ = ["v20_metrics_to_transfer", "select_provisional_keeper"]
