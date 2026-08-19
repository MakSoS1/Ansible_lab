from __future__ import annotations

import copy

from ecup_matching.ml.v21_promotion import evaluate_transfer_fold, evaluate_two_fold_transfer


def _metrics(*, human: float, weak: float, brier: float, tail: float) -> dict[str, object]:
    return {
        "human_macro_average_precision": human,
        "weak_macro_average_precision": weak,
        "weak_soft_brier": brier,
        "tail_macro_average_precision": tail,
        "per_category_ap": {"A": human, "B": human - 0.01},
        "category_row_counts": {"A": 500, "B": 400},
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "cross_split_item_overlap": 0,
    }


def _calibration(*, implied: float = 0.515, normalized: float = 3.6) -> dict[str, object]:
    return {
        "calibrated": True,
        "best_anchor": "v19",
        "candidate_above_v19": True,
        "normalized_gap_above_v19": normalized,
        "required_normalized_gap_for_target": 3.033065165249539,
        "proxy_implied_public_lb": implied,
        "target_public_lb": 0.5,
        "target_reached": implied >= 0.5 and normalized >= 3.033065165249539,
    }


def test_transfer_fold_combines_v19_retention_with_v21_public_gate() -> None:
    control = _metrics(human=0.700, weak=0.650, brier=0.110, tail=0.680)
    candidate = _metrics(human=0.701, weak=0.661, brier=0.109, tail=0.684)
    result = evaluate_transfer_fold(control, candidate, _calibration())
    assert result["human_delta"] > 0
    assert result["weak_delta"] > 0.005
    assert result["public_target_gate"] is True
    assert result["promote"] is True


def test_tiny_proxy_gain_can_never_be_called_target_half() -> None:
    control = _metrics(human=0.700, weak=0.650, brier=0.110, tail=0.680)
    candidate = _metrics(human=0.702, weak=0.665, brier=0.109, tail=0.685)
    result = evaluate_transfer_fold(control, candidate, _calibration(implied=0.43, normalized=0.7))
    assert result["public_target_gate"] is False
    assert result["promote"] is False


def test_weak_forgetting_or_human_regression_fails_closed() -> None:
    control = _metrics(human=0.700, weak=0.650, brier=0.110, tail=0.680)
    weak_bad = _metrics(human=0.701, weak=0.652, brier=0.109, tail=0.684)
    assert evaluate_transfer_fold(control, weak_bad, _calibration())["promote"] is False

    human_bad = _metrics(human=0.697, weak=0.665, brier=0.109, tail=0.684)
    assert evaluate_transfer_fold(control, human_bad, _calibration())["promote"] is False


def test_gold_or_overlap_provenance_violation_raises() -> None:
    control = _metrics(human=0.700, weak=0.650, brier=0.110, tail=0.680)
    candidate = _metrics(human=0.701, weak=0.661, brier=0.109, tail=0.684)
    bad = copy.deepcopy(candidate)
    bad["gold_metric_opened"] = True
    try:
        evaluate_transfer_fold(control, bad, _calibration())
    except ValueError:
        pass
    else:
        raise AssertionError("gold provenance violation must fail closed")


def test_two_fold_requires_each_fold_and_nonnegative_mean_human() -> None:
    c0 = _metrics(human=0.700, weak=0.650, brier=0.110, tail=0.680)
    c1 = _metrics(human=0.690, weak=0.645, brier=0.112, tail=0.670)
    a0 = _metrics(human=0.701, weak=0.661, brier=0.109, tail=0.684)
    a1 = _metrics(human=0.691, weak=0.656, brier=0.111, tail=0.674)
    result = evaluate_two_fold_transfer(c0, a0, _calibration(0.515, 3.6), c1, a1, _calibration(0.507, 3.3))
    assert result["mean_human_delta"] >= 0
    assert result["mean_weak_delta"] > 0.005
    assert result["min_proxy_implied_public_lb"] >= 0.5
    assert result["promote"] is True
