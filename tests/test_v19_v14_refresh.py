from __future__ import annotations

import pandas as pd

from ecup_matching.ml.v19_v14_refresh import filter_refresh_pairs, select_refresh_keeper


def test_filter_refresh_pairs_removes_seen_endpoints_and_keeps_soft_targets():
    seen = pd.DataFrame({"id1": [1, 3], "id2": [2, 4], "target": [0.99, 0.01]})
    candidate = pd.DataFrame(
        {
            "id1": [1, 5, 7, 9],
            "id2": [6, 8, 10, 11],
            "target": [0.95, 0.91, 0.08, 0.73],
            "weak_weight": [1.0, 0.6, 0.6, 0.3],
            "category": ["A", "A", "B", "B"],
        }
    )
    out, report = filter_refresh_pairs(candidate, seen)
    assert out[["id1", "id2"]].values.tolist() == [[5, 8], [7, 10], [9, 11]]
    assert out["target"].tolist() == [0.91, 0.08, 0.73]
    assert report["seen_endpoint_count"] == 4
    assert report["removed_seen_endpoint_rows"] == 1


def test_select_refresh_keeper_requires_weak_gain_brier_and_human_retention():
    base = {
        "human_macro_average_precision": 0.710,
        "weak_macro_average_precision": 0.650,
        "weak_soft_brier": 0.100,
        "per_category_ap": {"A": 0.70, "B": 0.80},
        "category_row_counts": {"A": 500, "B": 500},
        "gold_metric_opened": False,
        "cross_split_item_overlap": 0,
    }
    good = {
        "human_macro_average_precision": 0.7105,
        "weak_macro_average_precision": 0.658,
        "weak_soft_brier": 0.099,
        "per_category_ap": {"A": 0.699, "B": 0.802},
        "category_row_counts": {"A": 500, "B": 500},
        "gold_metric_opened": False,
        "cross_split_item_overlap": 0,
    }
    bad = dict(good)
    bad["weak_macro_average_precision"] = 0.654
    result = select_refresh_keeper(base, {"v18-ema": good, "v19-refresh": bad})
    assert result["keeper"] == "v18-ema"
    assert result["evaluations"]["v18-ema"]["promote"] is True
    assert result["evaluations"]["v19-refresh"]["promote"] is False


def test_select_refresh_keeper_prefers_higher_weak_gain_then_human_retention():
    base = {
        "human_macro_average_precision": 0.710,
        "weak_macro_average_precision": 0.650,
        "weak_soft_brier": 0.100,
        "per_category_ap": {"A": 0.70},
        "category_row_counts": {"A": 500},
        "gold_metric_opened": False,
        "cross_split_item_overlap": 0,
    }
    a = {
        "human_macro_average_precision": 0.710,
        "weak_macro_average_precision": 0.658,
        "weak_soft_brier": 0.099,
        "per_category_ap": {"A": 0.70},
        "category_row_counts": {"A": 500},
        "gold_metric_opened": False,
        "cross_split_item_overlap": 0,
    }
    b = dict(a)
    b["weak_macro_average_precision"] = 0.662
    result = select_refresh_keeper(base, {"v18-ema": a, "v19-refresh": b})
    assert result["keeper"] == "v19-refresh"
