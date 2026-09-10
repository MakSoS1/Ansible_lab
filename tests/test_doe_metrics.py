import numpy as np

from aios_track2.doe import generate_sobol_trajectories
from aios_track2.metrics import interval_coverage, ranking_metrics, rollout_error_by_horizon


def test_sobol_trajectories_are_reproducible_and_smooth() -> None:
    a = generate_sobol_trajectories(8, 6, 4, seed=19, max_delta=0.12)
    b = generate_sobol_trajectories(8, 6, 4, seed=19, max_delta=0.12)
    np.testing.assert_allclose(a, b)
    assert a.shape == (8, 6, 4)
    assert np.max(np.abs(np.diff(a, axis=1))) <= 0.1200001
    assert np.min(a) >= 0 and np.max(a) <= 1


def test_ranking_metrics_reward_correct_candidate_order() -> None:
    truth = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    pred = np.array([1.1, 3.1, 2.1, 5.2, 3.9])
    metrics = ranking_metrics(truth, pred, top_k=2)
    assert metrics["spearman"] > 0.9
    assert metrics["kendall"] > 0.8
    assert metrics["top_k_recall"] == 1.0
    assert metrics["pairwise_accuracy"] > 0.9


def test_interval_coverage_and_rollout_error_are_horizon_aware() -> None:
    y = np.array([[[1.0], [2.0], [3.0]], [[2.0], [4.0], [6.0]]])
    pred = y + np.array([[[0.1], [0.2], [0.8]], [[0.1], [0.2], [0.8]]])
    coverage = interval_coverage(y, pred - 0.5, pred + 0.5)
    errors = rollout_error_by_horizon(y, pred)
    assert 0 < coverage < 1
    np.testing.assert_allclose(errors, [0.1, 0.2, 0.8])
