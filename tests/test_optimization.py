from aios_track2.optimization import OptimizationRequest, ToyOptimizer, ToySurrogate, optimize, risk_score


def test_ood_candidate_cannot_win_without_promotion() -> None:
    result = ToyOptimizer(seed=3).run()
    assert result.best.accepted is True
    assert result.best.opm_validated is True


def test_risk_adjusted_score_penalizes_uncertainty() -> None:
    assert risk_score(mean_npv=100.0, std_npv=20.0, penalty=1.0) == 80.0


def test_cem_and_cma_converge() -> None:
    request = OptimizationRequest(seed=0, population=12, iterations=5, elites=3, method="cem")
    cem = optimize(request, surrogate=ToySurrogate())
    cma = optimize(
        OptimizationRequest(seed=1, population=12, iterations=5, elites=3, method="cma"),
        surrogate=ToySurrogate(),
    )
    assert cem.best.mean_npv > 50
    assert cma.best.mean_npv > 50
