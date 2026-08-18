from __future__ import annotations

from ecup_matching.ml.v19_refresh_gate import evaluate_refresh, evaluate_two_fold_refresh


def _metrics(human: float, weak: float, brier: float, *, a: float = 0.8, b: float = 0.7) -> dict:
    return {
        "human_macro_average_precision": human,
        "weak_macro_average_precision": weak,
        "weak_soft_brier": brier,
        "per_category_ap": {"a": a, "b": b},
        "category_row_counts": {"a": 500, "b": 300},
        "gold_metric_opened": False,
        "cross_split_item_overlap": 0,
    }


def test_refresh_requires_strict_weak_gain() -> None:
    pre = _metrics(0.700, 0.650, 0.20)
    boundary = _metrics(0.700, 0.655, 0.20)
    winner = _metrics(0.699, 0.65501, 0.20)
    assert evaluate_refresh(pre, boundary)["promote"] is False
    assert evaluate_refresh(pre, winner)["promote"] is True


def test_refresh_rejects_human_tail_and_brier_regression() -> None:
    pre = _metrics(0.700, 0.650, 0.20)
    human_bad = _metrics(0.6979, 0.660, 0.20)
    tail_bad = _metrics(0.700, 0.660, 0.20, b=0.669)
    brier_bad = _metrics(0.700, 0.660, 0.20201)
    assert evaluate_refresh(pre, human_bad)["promote"] is False
    assert evaluate_refresh(pre, tail_bad)["promote"] is False
    assert evaluate_refresh(pre, brier_bad)["promote"] is False


def test_two_fold_confirmation_requires_each_fold_and_nonnegative_mean_human() -> None:
    pre0 = _metrics(0.700, 0.650, 0.20)
    post0 = _metrics(0.701, 0.658, 0.199)
    pre1 = _metrics(0.690, 0.640, 0.21)
    post1 = _metrics(0.690, 0.647, 0.209)
    result = evaluate_two_fold_refresh(pre0, post0, pre1, post1)
    assert result["promote"] is True
    post1_bad = _metrics(0.6880, 0.647, 0.209)
    assert evaluate_two_fold_refresh(pre0, post0, pre1, post1_bad)["promote"] is False
