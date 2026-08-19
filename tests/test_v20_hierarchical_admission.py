from __future__ import annotations

import pandas as pd

from ecup_matching.ml.v20_admission import (
    build_hierarchical_policy,
    row_passes_hierarchical_policy,
    wilson_lower_bound,
)
from ecup_matching.ml.v20_policy import V20Policy


def _rows(n: int, *, truth: int, pred: int, reason: str, category: str = "phones") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "truth": [truth] * n,
            "pred": [pred] * n,
            "reason_code": [reason] * n,
            "category": [category] * n,
        }
    )


def test_100_of_100_cannot_satisfy_negative_0995_wilson_floor():
    assert wilson_lower_bound(100, 100) < V20Policy().negative_precision_lcb


def test_pooled_reason_support_can_pass_without_lowering_floors():
    audit = pd.concat(
        [
            _rows(900, truth=0, pred=0, reason="MODEL_CONFLICT", category="phones"),
            _rows(900, truth=0, pred=0, reason="MODEL_CONFLICT", category="laptops"),
            _rows(900, truth=1, pred=1, reason="SAME_MODEL", category="phones"),
            _rows(900, truth=1, pred=1, reason="SAME_MODEL", category="laptops"),
        ],
        ignore_index=True,
    )
    report = build_hierarchical_policy(audit, V20Policy())

    assert report["version"] == "v20-hierarchical-admission-v2"
    assert report["predicted_labels"]["0"]["lcb"] >= V20Policy().negative_precision_lcb
    assert report["predicted_labels"]["1"]["lcb"] >= V20Policy().positive_precision_lcb
    assert report["reason_labels"]["MODEL_CONFLICT"]["0"]["pass"] is True
    assert report["reason_diagnostics"]["MODEL_CONFLICT"]["pass_any_label"] is True
    assert report["categories"]["phones"]["pass"] is True


def test_row_requires_all_applicable_hierarchical_gates():
    audit = pd.concat(
        [
            _rows(1800, truth=0, pred=0, reason="MODEL_CONFLICT", category="phones"),
            _rows(1800, truth=1, pred=1, reason="SAME_MODEL", category="phones"),
        ],
        ignore_index=True,
    )
    report = build_hierarchical_policy(audit, V20Policy())
    row = {"pred": 0, "reason_code": "MODEL_CONFLICT", "category": "phones"}
    assert row_passes_hierarchical_policy(row, report) is True

    broken = dict(report)
    broken["critical_family"] = dict(report["critical_family"])
    broken["critical_family"]["pass"] = False
    assert row_passes_hierarchical_policy(row, broken) is False


def test_unknown_reason_fails_closed():
    audit = pd.concat(
        [
            _rows(1800, truth=0, pred=0, reason="MODEL_CONFLICT", category="phones"),
            _rows(1800, truth=1, pred=1, reason="SAME_MODEL", category="phones"),
        ],
        ignore_index=True,
    )
    report = build_hierarchical_policy(audit, V20Policy())
    row = {"pred": 0, "reason_code": "OTHER", "category": "phones"}
    assert row_passes_hierarchical_policy(row, report) is False
