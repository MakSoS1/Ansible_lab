from __future__ import annotations

import pandas as pd

from ecup_matching.ml.run_v20_filter_candidate_queue import filter_candidate_queue


def _policy(*, model_pass: bool = True, phone_pass: bool = True, critical_pass: bool = True) -> dict:
    return {
        "version": "v20-admission-policy-v2",
        "hierarchical": {
            "version": "v20-hierarchical-admission-v1",
            "reasons": {
                "MODEL_CONFLICT": {"pass": model_pass, "lcb": 0.997},
                "SAME_MODEL": {"pass": True, "lcb": 0.990},
            },
            "categories": {
                "phones": {"pass": phone_pass, "lcb": 0.980},
                "laptops": {"pass": True, "lcb": 0.980},
            },
            "critical_family": {"pass": critical_pass, "lcb": 0.960},
        },
    }


def test_queue_requires_reason_and_category_to_pass_both_folds():
    candidates = pd.DataFrame(
        [
            {"id1": 1, "id2": 2, "category": "phones", "reason_code": "MODEL_CONFLICT", "stratum": "phones|MODEL_CONFLICT|hard"},
            {"id1": 3, "id2": 4, "category": "laptops", "reason_code": "SAME_MODEL", "stratum": "laptops|SAME_MODEL|easy"},
        ]
    )
    out, report = filter_candidate_queue(candidates, [_policy(), _policy(phone_pass=False)])
    assert list(out[["id1", "id2"]].itertuples(index=False, name=None)) == [(3, 4)]
    assert report["input_rows"] == 2
    assert report["output_rows"] == 1


def test_critical_candidate_requires_critical_family_on_both_folds():
    candidates = pd.DataFrame(
        [{"id1": 1, "id2": 2, "category": "phones", "reason_code": "MODEL_CONFLICT", "stratum": "phones|MODEL_CONFLICT|hard"}]
    )
    out, _ = filter_candidate_queue(candidates, [_policy(), _policy(critical_pass=False)])
    assert out.empty


def test_queue_remains_target_free():
    candidates = pd.DataFrame(
        [{"id1": 3, "id2": 4, "category": "laptops", "reason_code": "SAME_MODEL", "stratum": "laptops|SAME_MODEL|easy"}]
    )
    out, report = filter_candidate_queue(candidates, [_policy(), _policy()])
    assert "target" not in out.columns
    assert report["target_column_present"] is False


def test_target_bearing_candidate_input_is_rejected():
    candidates = pd.DataFrame(
        [{"id1": 1, "id2": 2, "target": 0, "category": "phones", "reason_code": "MODEL_CONFLICT", "stratum": "x"}]
    )
    try:
        filter_candidate_queue(candidates, [_policy(), _policy()])
    except ValueError as exc:
        assert "target-free" in str(exc)
    else:
        raise AssertionError("target-bearing candidate input must fail")
