import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.v8_distribution import (
    binary_prevalence,
    candidate_graph_summary,
    category_distribution_report,
    human_excluded_llm_mask,
    target_distribution,
)


def test_binary_prevalence_accepts_only_binary_targets():
    frame = pd.DataFrame({"target": [1, 0, 1, 0, 0]})
    assert binary_prevalence(frame) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="binary"):
        binary_prevalence(pd.DataFrame({"target": [0.0, 0.25, 1.0]}))


def test_target_distribution_reports_soft_label_shape_without_calling_it_prevalence():
    frame = pd.DataFrame({"target": [0.0, 0.01, 0.5, 0.99, 1.0]})
    report = target_distribution(frame)
    assert report["rows"] == 5
    assert report["mean"] == pytest.approx(0.5)
    assert report["frac_lt_005"] == pytest.approx(0.4)
    assert report["frac_gt_095"] == pytest.approx(0.4)
    assert report["frac_020_080"] == pytest.approx(0.2)
    assert report["is_binary"] is False
    assert "binary_prevalence" not in report


def test_candidate_graph_summary_counts_endpoint_degrees():
    frame = pd.DataFrame({"id1": [1, 1, 2], "id2": [2, 3, 4]})
    report = candidate_graph_summary(frame)
    assert report["rows"] == 3
    assert report["unique_items"] == 4
    assert report["mean_endpoint_degree"] == pytest.approx(1.5)
    assert report["median_endpoint_degree"] == pytest.approx(1.5)
    assert report["max_endpoint_degree"] == 2
    assert report["rows_per_unique_item"] == pytest.approx(0.75)


def test_human_excluded_llm_mask_removes_pair_when_either_endpoint_is_human():
    llm = pd.DataFrame({"id1": [1, 4, 5, 8], "id2": [2, 5, 7, 9]})
    mask = human_excluded_llm_mask(llm, {1, 7, 9})
    assert mask.dtype == np.bool_
    assert mask.tolist() == [False, True, False, False]


def test_category_distribution_report_requires_same_category_on_both_endpoints():
    pairs = pd.DataFrame(
        {
            "id1": [1, 3, 5, 7],
            "id2": [2, 4, 6, 8],
            "target": [1.0, 0.0, 0.9, 0.1],
        }
    )
    categories = {1: "A", 2: "A", 3: "A", 4: "A", 5: "B", 6: "B", 7: "B", 8: "B"}
    report = category_distribution_report(pairs, categories)
    assert report["A"]["rows"] == 2
    assert report["A"]["target_mean"] == pytest.approx(0.5)
    assert report["A"]["is_binary"] is True
    assert report["A"]["binary_prevalence"] == pytest.approx(0.5)
    assert report["B"]["rows"] == 2
    assert report["B"]["is_binary"] is False
    assert "binary_prevalence" not in report["B"]

    bad = pairs.iloc[[0]].copy()
    with pytest.raises(ValueError, match="category"):
        category_distribution_report(bad, {1: "A", 2: "B"})
