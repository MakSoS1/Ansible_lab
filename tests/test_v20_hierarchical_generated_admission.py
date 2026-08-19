from __future__ import annotations

import pandas as pd

from ecup_matching.ml.v20_admission import (
    build_hierarchical_policy,
    hierarchical_reliability,
    row_passes_hierarchical_policy,
)
from ecup_matching.ml.v20_policy import V20Policy


def _audit() -> pd.DataFrame:
    rows = []
    for category in ("phones", "laptops"):
        for i in range(1000):
            rows.append({"truth": 0, "pred": 0, "reason_code": "MODEL_CONFLICT", "category": category})
            rows.append({"truth": 1, "pred": 1, "reason_code": "SAME_MODEL", "category": category})
    return pd.DataFrame(rows)


def test_candidate_reliability_is_minimum_of_all_applicable_gates():
    report = build_hierarchical_policy(_audit(), V20Policy())
    row = {"target": 0, "reason_code": "MODEL_CONFLICT", "category": "phones"}
    assert row_passes_hierarchical_policy(row, report) is True
    expected = min(
        report["predicted_labels"]["0"]["lcb"],
        report["reasons"]["MODEL_CONFLICT"]["lcb"],
        report["categories"]["phones"]["lcb"],
        report["critical_family"]["lcb"],
    )
    assert hierarchical_reliability(row, report) == expected


def test_reliability_is_zero_when_any_gate_fails():
    report = build_hierarchical_policy(_audit(), V20Policy())
    broken = dict(report)
    broken["reasons"] = dict(report["reasons"])
    broken["reasons"]["MODEL_CONFLICT"] = dict(report["reasons"]["MODEL_CONFLICT"])
    broken["reasons"]["MODEL_CONFLICT"]["pass"] = False
    row = {"target": 0, "reason_code": "MODEL_CONFLICT", "category": "phones"}
    assert hierarchical_reliability(row, broken) == 0.0


def test_noncritical_candidate_does_not_depend_on_critical_gate():
    report = build_hierarchical_policy(_audit(), V20Policy())
    broken = dict(report)
    broken["critical_family"] = dict(report["critical_family"])
    broken["critical_family"]["pass"] = False
    row = {"target": 1, "reason_code": "SAME_MODEL", "category": "phones"}
    assert hierarchical_reliability(row, broken) > 0.0
