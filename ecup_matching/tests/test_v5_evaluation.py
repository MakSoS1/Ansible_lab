import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.v5_evaluation import (
    assert_gold_evaluation_eligible,
    candidate_freeze_hash,
    macro_ap_report,
    paired_component_bootstrap,
)


def test_macro_ap_report_returns_per_category_and_unweighted_macro():
    frame = pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 1, 0, 0, 1],
            "category": ["a"] * 4 + ["b"] * 4,
        }
    )
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.6, 0.2, 0.9])
    report = macro_ap_report(frame, scores)

    assert set(report["per_category_ap"]) == {"a", "b"}
    expected = np.mean(list(report["per_category_ap"].values()))
    assert report["macro_average_precision"] == pytest.approx(expected)
    assert report["macro_average_precision"] == pytest.approx(1.0)


def test_macro_ap_report_strict_official_rejects_toy_category_set():
    frame = pd.DataFrame({"target": [0, 1], "category": ["a", "a"]})
    with pytest.raises(ValueError, match="category set"):
        macro_ap_report(frame, np.array([0.1, 0.9]), strict_official=True)


def test_gold_evaluator_requires_frozen_matching_hashes():
    freeze = {
        "frozen": True,
        "split_sha": "split-123",
        "config_sha": "config-456",
        "prediction_sha": "pred-789",
    }
    assert_gold_evaluation_eligible(
        freeze,
        split_sha="split-123",
        config_sha="config-456",
        prediction_sha="pred-789",
    )

    with pytest.raises(ValueError, match="frozen"):
        assert_gold_evaluation_eligible(
            {**freeze, "frozen": False},
            split_sha="split-123",
            config_sha="config-456",
            prediction_sha="pred-789",
        )
    with pytest.raises(ValueError, match="config_sha"):
        assert_gold_evaluation_eligible(
            freeze,
            split_sha="split-123",
            config_sha="different",
            prediction_sha="pred-789",
        )


def test_candidate_freeze_hash_is_deterministic_and_config_sensitive():
    first = candidate_freeze_hash(
        {"encoder": "tiny", "residual": 0.1},
        prediction_sha="pred",
        split_sha="split",
    )
    second = candidate_freeze_hash(
        {"residual": 0.1, "encoder": "tiny"},
        prediction_sha="pred",
        split_sha="split",
    )
    changed = candidate_freeze_hash(
        {"encoder": "tiny", "residual": 0.2},
        prediction_sha="pred",
        split_sha="split",
    )
    assert first == second
    assert first != changed


def test_component_bootstrap_is_paired_deterministic_and_reports_delta_interval():
    frame = pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 1, 0, 1, 0] * 2,
            "category": ["a"] * 8 + ["b"] * 8,
            "component": np.repeat(np.arange(8), 2),
        }
    )
    base = np.array([0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45] * 2)
    improved = np.array([0.95, 0.05, 0.9, 0.1, 0.85, 0.15, 0.8, 0.2] * 2)

    report = paired_component_bootstrap(
        frame,
        base,
        improved,
        component_col="component",
        n_bootstrap=100,
        seed=2026,
    )
    report2 = paired_component_bootstrap(
        frame,
        base,
        improved,
        component_col="component",
        n_bootstrap=100,
        seed=2026,
    )

    assert report == report2
    assert report["point_delta"] >= 0.0
    assert report["ci_low"] <= report["median_delta"] <= report["ci_high"]
