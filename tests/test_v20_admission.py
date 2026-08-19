import pandas as pd

from ecup_matching.ml.v20_admission import (
    admit_strata,
    build_fold_safe_audit_split,
    wilson_lower_bound,
)
from ecup_matching.ml.v20_policy import V20Policy


def test_wilson_bounds_are_sane():
    assert wilson_lower_bound(0, 0) == 0.0
    assert 0.95 < wilson_lower_bound(1000, 1000) < 1.0
    assert wilson_lower_bound(990, 1000) < 0.99


def test_admission_rejects_insufficient_support_and_respects_label_floor():
    policy = V20Policy(min_stratum_support=100)
    rows = pd.DataFrame([
        {"stratum": "A", "truth": 1, "pred": 1, "reason_code": "SAME_MODEL"}
        for _ in range(99)
    ])
    report = admit_strata(rows, policy)
    assert report["strata"]["A"]["admitted"] is False
    assert report["strata"]["A"]["reason"] == "insufficient_support"


def test_perfect_supported_positive_stratum_is_admitted():
    policy = V20Policy(min_stratum_support=100)
    rows = pd.DataFrame([
        {"stratum": "A", "truth": 1, "pred": 1, "reason_code": "SAME_MODEL"}
        for _ in range(1000)
    ])
    report = admit_strata(rows, policy)
    assert report["strata"]["A"]["admitted"] is True


def test_fold_safe_audit_split_has_no_item_overlap():
    frame = pd.DataFrame({
        "id1": [1, 3, 5, 7, 9, 11],
        "id2": [2, 4, 6, 8, 10, 12],
        "target": [1, 0, 1, 0, 1, 0],
        "category": ["x"] * 6,
        "stratum": ["s"] * 6,
    })
    train, audit, report = build_fold_safe_audit_split(frame, audit_fraction=0.33, seed=2026)
    train_items = set(train.id1) | set(train.id2)
    audit_items = set(audit.id1) | set(audit.id2)
    assert not (train_items & audit_items)
    assert report["item_overlap"] == 0
