import pandas as pd

from ecup_matching.ml.v20_admission import (
    build_hierarchical_policy,
    hierarchical_reliability,
    row_passes_hierarchical_policy,
)
from ecup_matching.ml.v20_policy import V20Policy


def test_same_reason_can_admit_both_labels_when_each_bucket_is_reliable():
    rows = []
    for _ in range(1200):
        rows.append({"truth": 1, "pred": 1, "reason_code": "OTHER", "category": "phones"})
        rows.append({"truth": 0, "pred": 0, "reason_code": "OTHER", "category": "phones"})
    frame = pd.DataFrame(rows)

    report = build_hierarchical_policy(frame, V20Policy(min_stratum_support=100))

    assert report["version"] == "v20-hierarchical-admission-v2"
    assert report["reason_labels"]["OTHER"]["1"]["pass"] is True
    assert report["reason_labels"]["OTHER"]["0"]["pass"] is True

    positive = {"pred": 1, "reason_code": "OTHER", "category": "phones"}
    negative = {"pred": 0, "reason_code": "OTHER", "category": "phones"}
    assert row_passes_hierarchical_policy(positive, report) is True
    assert row_passes_hierarchical_policy(negative, report) is True
    assert hierarchical_reliability(positive, report) >= 0.985
    assert hierarchical_reliability(negative, report) >= 0.995
