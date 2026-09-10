import numpy as np

from aios_track2.optimization import diagonal_cma_es, risk_aware_cem
from aios_track2.strategies import TOP5_STRATEGIES, strategy_names


def sphere_objective(pop: np.ndarray):
    target = np.array([0.2, 0.7, 0.4])
    value = 10.0 - np.sum((pop - target) ** 2, axis=1)
    uncertainty = 0.1 * np.sum(np.abs(pop - 0.5), axis=1)
    valid = np.all((pop >= 0.05) & (pop <= 0.95), axis=1)
    return value, uncertainty, valid


def test_risk_aware_cem_improves_over_center() -> None:
    center = np.full((1, 3), 0.5)
    center_score = sphere_objective(center)[0][0]
    result = risk_aware_cem(
        sphere_objective,
        dim=3,
        seed=3,
        population=80,
        iterations=12,
        risk_beta=0.2,
    )
    assert result.score > center_score
    assert np.all((result.x >= 0.05) & (result.x <= 0.95))


def test_diagonal_cma_es_finds_high_value_feasible_region() -> None:
    result = diagonal_cma_es(
        sphere_objective,
        dim=3,
        seed=4,
        population=64,
        iterations=16,
        risk_beta=0.1,
    )
    assert result.score > 9.85
    assert np.all((result.x >= 0.05) & (result.x <= 0.95))


def test_top_five_strategy_portfolio_is_explicit_and_unique() -> None:
    assert strategy_names() == (
        "linear_local",
        "gru_cem",
        "tcn_cma",
        "graph_risk_cem",
        "graph_mappo",
    )
    assert len(TOP5_STRATEGIES) == 5
    assert len(set(strategy_names())) == 5
    primary = next(x for x in TOP5_STRATEGIES if x.name == "graph_risk_cem")
    assert primary.uncertainty_aware is True
    assert primary.active_learning is True
