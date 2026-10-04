import numpy as np
import pandas as pd

from aios_track2.active_learning import acquisition_scores, select_for_opm
from aios_track2.physics import (
    compensation_ratio,
    detect_water_breakthrough,
    lagged_connectivity,
    physical_consistency_metrics,
)


def test_compensation_ratio_and_water_breakthrough() -> None:
    assert compensation_ratio(injection=100, withdrawal=80) == 1.25
    assert np.isnan(compensation_ratio(injection=100, withdrawal=0))
    assert detect_water_breakthrough(np.array([0.10, 0.12, 0.14, 0.35, 0.37])) == 3
    assert detect_water_breakthrough(np.array([0.10, 0.12, 0.14])) is None


def test_lagged_connectivity_recovers_delayed_response() -> None:
    injection = np.sin(np.linspace(0, 6 * np.pi, 80))
    response = np.concatenate([np.zeros(4), injection[:-4]])
    graph = lagged_connectivity({"I1": injection}, {"P1": response}, max_lag=10)
    edge = graph[("I1", "P1")]
    assert edge.lag == 4
    assert edge.correlation > 0.99


def test_physical_metrics_count_constraint_failures() -> None:
    frame = pd.DataFrame(
        {
            "withdrawal": [100.0, 100.0],
            "injection": [100.0, 60.0],
            "BHP": [250.0, 320.0],
            "WCT": [0.5, 1.2],
            "WLPR": [400.0, 510.0],
        }
    )
    metrics = physical_consistency_metrics(frame, max_bhp=300)
    assert metrics["pressure_violation_rate"] == 0.5
    assert metrics["watercut_violation_rate"] == 0.5
    assert metrics["wlpr_violation_rate"] == 0.5


def test_active_learning_combines_value_uncertainty_and_novelty() -> None:
    training = np.array([[0.1, 0.1], [0.5, 0.5]])
    candidates = np.array([[0.51, 0.51], [0.9, 0.9], [0.2, 0.8]])
    npv = np.array([10.0, 10.2, 9.9])
    uncertainty = np.array([0.01, 0.20, 0.30])
    scores = acquisition_scores(candidates, training, npv, uncertainty)
    selected = select_for_opm(scores, budget=2)
    assert len(selected) == 2
    assert 0 not in set(selected)
