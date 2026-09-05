import numpy as np

from aios_track2.metrics import evaluate_surrogate
from aios_track2.quality_gate import evaluate_quality_gate


def test_surrogate_metrics_expose_r2_quality_gate() -> None:
    y = np.arange(20, dtype=float).reshape(2, 5, 2)
    result = evaluate_surrogate(y, y + 0.01)
    assert result["r2"] > 0.95


def test_quality_gate_requires_independent_095_metrics_and_zero_physics() -> None:
    report = evaluate_quality_gate(
        dynamic={"r2": 0.97, "nrmse": 0.03},
        ranking={"spearman": 0.96, "pairwise_accuracy": 0.96, "top_k_recall": 1.0},
        physics_violation_rate=0.0,
    )
    assert report.passed
    assert report.minimum_quality_metric == 0.96


def test_quality_gate_rejects_holdout_top3_recall_two_of_three() -> None:
    report = evaluate_quality_gate(
        dynamic={"r2": 0.9512247034582642, "nrmse": 0.027523284453274358},
        ranking={"spearman": 0.9911764705882352, "pairwise_accuracy": 0.975, "top_k_recall": 2 / 3},
        physics_violation_rate=0.0,
    )
    assert not report.passed
    assert "TOPK_LT_090" in report.failures


def test_quality_gate_cannot_hide_bad_ranking_behind_good_r2() -> None:
    report = evaluate_quality_gate(
        dynamic={"r2": 0.995, "nrmse": 0.01},
        ranking={"spearman": 0.8, "pairwise_accuracy": 0.99, "top_k_recall": 1.0},
        physics_violation_rate=0.0,
    )
    assert not report.passed
    assert "SPEARMAN_LT_095" in report.failures
