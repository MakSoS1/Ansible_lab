import numpy as np

from aios_track2.bakeoff import CandidateEvaluation, choose_competition_winner
from aios_track2.surrogates.gru import GRUSurrogate
from aios_track2.training import fit_torch_surrogate
from aios_track2.validation import quality_gate


def test_short_training_loop_reduces_validation_loss() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 6, 3, 2)).astype("float32")
    y = (0.6 * x[..., :1] - 0.2 * x[..., 1:2]).astype("float32")
    model = GRUSurrogate(2, 1, hidden=8)
    report = fit_torch_surrogate(
        model,
        x[:9],
        y[:9],
        x[9:],
        y[9:],
        epochs=25,
        lr=0.01,
        seed=11,
    )
    assert report.validation_loss[-1] < report.validation_loss[0]


def test_quality_gate_is_multi_metric_not_single_npv() -> None:
    good = quality_gate(
        {
            "spearman": 0.9,
            "top_k_recall": 0.8,
            "nrmse": 0.08,
            "coverage_90": 0.88,
            "constraint_violation_rate": 0.0,
            "opm_npv_regret": 0.04,
        }
    )
    bad = quality_gate(
        {
            "spearman": 0.9,
            "top_k_recall": 0.8,
            "nrmse": 0.08,
            "coverage_90": 0.20,
            "constraint_violation_rate": 0.0,
            "opm_npv_regret": 0.04,
        }
    )
    assert good.passed is True
    assert bad.passed is False
    assert any("coverage_90" in failure for failure in bad.failures)


def test_bakeoff_winner_requires_hard_gates_then_maximizes_opm_npv() -> None:
    rows = [
        CandidateEvaluation("high_invalid", 15.0, False, 0, 20, 0.9),
        CandidateEvaluation("valid_a", 12.0, True, 0, 40, 0.8),
        CandidateEvaluation("valid_b", 12.5, True, 0, 60, 0.75),
        CandidateEvaluation("violating", 14.0, True, 1, 10, 0.99),
    ]
    winner = choose_competition_winner(rows)
    assert winner.name == "valid_b"
