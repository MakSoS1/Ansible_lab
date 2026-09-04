from __future__ import annotations

import numpy as np
import pytest

from aios_track2.real_validation import (
    RbfKernelRidge,
    align_common_post_start,
    dynamic_delta_report,
    ranking_report,
)


def test_align_common_post_start_excludes_history_and_uses_intersection() -> None:
    dates = (
        np.asarray(["2006-12-01", "2007-01-01", "2007-02-01", "2007-03-01"]),
        np.asarray(["2006-11-01", "2007-01-01", "2007-03-01", "2007-04-01"]),
    )
    common = align_common_post_start(dates, start_date="2007-01-01")
    assert common.tolist() == ["2007-01-01", "2007-03-01"]


def test_dynamic_delta_report_separates_aggregate_from_tail() -> None:
    baseline = np.zeros((4, 2), dtype=float)
    truth = np.stack(
        [
            baseline,
            np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        ]
    )
    prediction = truth.copy()
    prediction[2, :, 1] = np.asarray([0.0, 6.0, 0.0, 6.0])

    report = dynamic_delta_report(
        truth,
        prediction,
        baseline=baseline,
        scenario_ids=(1, 2),
        channels=("oil", "pressure"),
    )

    assert report["aggregate_channel_r2"]["oil"] == pytest.approx(1.0)
    assert report["worst_scenario_channel"]["scenario_id"] == 2
    assert report["worst_scenario_channel"]["channel"] == "pressure"
    assert report["worst_scenario_channel"]["r2"] < report["min_aggregate_channel_r2"]
    assert 0.0 <= report["p10_scenario_channel_r2"] <= 1.0


def test_ranking_report_contains_regret_and_topk() -> None:
    truth = np.asarray([100.0, 200.0, 150.0, 175.0])
    prediction = np.asarray([100.0, 190.0, 180.0, 170.0])
    report = ranking_report(truth, prediction, top_k=2)
    assert report["simple_regret"] == pytest.approx(0.0)
    assert report["top_k_recall"] == pytest.approx(0.5)
    assert report["mae"] == pytest.approx(16.25)
    assert report["max_abs_error"] == pytest.approx(30.0)
    assert report["pairwise_accuracy"] < 1.0


def test_rbf_kernel_ridge_is_deterministic_and_vector_valued() -> None:
    x = np.asarray([[0.8], [1.0], [1.2]])
    y = np.asarray([[0.0, 0.0], [1.0, 2.0], [4.0, 8.0]])
    first = RbfKernelRidge(length_scale=2.0, ridge=1e-8, center=1.0, scale=0.2).fit(x, y)
    second = RbfKernelRidge(length_scale=2.0, ridge=1e-8, center=1.0, scale=0.2).fit(x, y)
    p1 = first.predict(x)
    p2 = second.predict(x)
    assert p1.shape == y.shape
    np.testing.assert_allclose(p1, p2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(p1, y, rtol=1e-5, atol=1e-5)
