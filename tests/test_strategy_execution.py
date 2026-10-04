import numpy as np

from aios_track2.strategies import strategy_names
from aios_track2.strategy_runner import run_blackbox_strategy


def objective(pop: np.ndarray):
    target = np.linspace(0.3, 0.7, pop.shape[1])
    value = 5.0 - np.mean((pop - target) ** 2, axis=1)
    uncertainty = 0.02 + 0.1 * np.mean(np.abs(pop - 0.5), axis=1)
    valid = np.all((pop >= 0.04) & (pop <= 0.96), axis=1)
    return value, uncertainty, valid


def test_all_five_strategies_execute_through_common_contract() -> None:
    runs = [
        run_blackbox_strategy(name, objective, dim=8, seed=100 + i, budget=128)
        for i, name in enumerate(strategy_names())
    ]
    assert [r.strategy for r in runs] == list(strategy_names())
    assert all(r.x.shape == (8,) for r in runs)
    assert all(np.isfinite(r.score) for r in runs)
    assert all(r.evaluations > 0 for r in runs)
    assert all(r.valid for r in runs)


def test_mappo_blackbox_training_uses_the_supplied_objective() -> None:
    def prefer_low(pop: np.ndarray):
        value = -np.mean(pop, axis=1)
        return value, np.zeros(len(pop)), np.ones(len(pop), dtype=bool)

    def prefer_high(pop: np.ndarray):
        value = np.mean(pop, axis=1)
        return value, np.zeros(len(pop)), np.ones(len(pop), dtype=bool)

    low = run_blackbox_strategy("graph_mappo", prefer_low, dim=6, seed=77, budget=256)
    high = run_blackbox_strategy("graph_mappo", prefer_high, dim=6, seed=77, budget=256)

    assert np.mean(low.x) < np.mean(high.x)
    assert low.score == np.max([-np.mean(low.x)])
    assert high.score == np.mean(high.x)
