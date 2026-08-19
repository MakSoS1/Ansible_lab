from __future__ import annotations

from ecup_matching.ml.v18_select import (
    evaluate_combination,
    evaluate_scaled_confirmation,
    evaluate_single,
    select_mechanisms,
)


def _metrics(human: float, weak: float, *, a: float = 0.80, b: float = 0.70) -> dict:
    return {
        "fold_macro_average_precision": human,
        "weak_holdout_after_human_phase": {"macro_average_precision": weak},
        "per_category_ap": {"a": a, "b": b},
        "category_row_counts": {"a": 500, "b": 300},
        "gold_metric_opened": False,
        "cross_split_item_overlap": 0,
    }


def test_single_requires_strict_weak_gain() -> None:
    control = _metrics(0.700, 0.400)
    boundary = _metrics(0.700, 0.403)
    winner = _metrics(0.699, 0.40301)
    assert evaluate_single(control, boundary)["promote"] is False
    assert evaluate_single(control, winner)["promote"] is True


def test_single_rejects_human_and_category_regression() -> None:
    control = _metrics(0.700, 0.400)
    human_bad = _metrics(0.6969, 0.410)
    tail_bad = _metrics(0.701, 0.410, b=0.669)
    assert evaluate_single(control, human_bad)["promote"] is False
    assert evaluate_single(control, tail_bad)["promote"] is False


def test_select_mechanisms_returns_only_independent_passes() -> None:
    control = _metrics(0.700, 0.400)
    candidates = {
        "quality": _metrics(0.7005, 0.406),
        "views": _metrics(0.695, 0.420),
        "ema": _metrics(0.701, 0.404),
    }
    selected = select_mechanisms(control, candidates)
    assert selected["selected"] == ["ema", "quality"]


def test_combination_must_beat_best_single_robust_score() -> None:
    control = _metrics(0.700, 0.400)
    singles = {
        "quality": _metrics(0.701, 0.407),
        "ema": _metrics(0.7005, 0.406),
    }
    not_better = _metrics(0.700, 0.4051)
    better = _metrics(0.702, 0.412)
    assert evaluate_combination(control, singles, not_better)["promote"] is False
    assert evaluate_combination(control, singles, better)["promote"] is True


def test_scaled_confirmation_uses_two_human_folds_and_weak_gate() -> None:
    c0 = _metrics(0.700, 0.400)
    c1 = _metrics(0.690, 0.395)
    q0 = _metrics(0.701, 0.407)
    q1 = _metrics(0.690, 0.402)
    result = evaluate_scaled_confirmation(c0, q0, c1, q1)
    assert result["promote"] is True
    q1_bad = _metrics(0.6879, 0.402)
    assert evaluate_scaled_confirmation(c0, q0, c1, q1_bad)["promote"] is False
