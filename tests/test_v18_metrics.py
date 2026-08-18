from __future__ import annotations

from ecup_matching.ml.v18_metrics import robust_gain_score, worst_qualifying_category_delta


def test_worst_category_delta_ignores_tiny_categories() -> None:
    result = worst_qualifying_category_delta(
        candidate_per_category={"a": 0.80, "b": 0.40, "tiny": 0.0},
        control_per_category={"a": 0.78, "b": 0.45, "tiny": 1.0},
        category_rows={"a": 500, "b": 250, "tiny": 30},
        min_rows=200,
    )
    assert result["qualifying_categories"] == 2
    assert result["worst_category"] == "b"
    assert abs(result["worst_delta"] - (-0.05)) < 1e-12


def test_robust_gain_rewards_two_axis_gain_and_penalizes_tail_regression() -> None:
    clean = robust_gain_score(human_delta=0.004, weak_delta=0.010, worst_category_delta=0.0)
    tail_bad = robust_gain_score(human_delta=0.004, weak_delta=0.010, worst_category_delta=-0.04)
    weaker = robust_gain_score(human_delta=0.001, weak_delta=0.004, worst_category_delta=0.0)
    assert clean > tail_bad
    assert clean > weaker
