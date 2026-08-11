import numpy as np
import pandas as pd

from ecup_matching.ml.v5_oof_aggregate import aggregate_oof_scores


def test_oof_aggregate_preserves_manifest_order_and_reports_fold_deltas():
    frame = pd.DataFrame(
        {
            "target": [1, 0, 1, 0] * 5,
            "category": ["a", "a", "b", "b"] * 5,
        }
    )
    folds = np.repeat(np.arange(5), 4)
    base = np.array([0.7, 0.3, 0.65, 0.35] * 5, dtype=float)
    candidate = np.array([0.9, 0.1, 0.85, 0.15] * 5, dtype=float)

    result = aggregate_oof_scores(frame, base, candidate, folds)

    assert result["macro_average_precision"] >= result["base_macro_average_precision"]
    assert len(result["fold_reports"]) == 5
    assert all(r["valid_rows"] == 4 for r in result["fold_reports"])
    assert all(r["delta_vs_base"] >= 0 for r in result["fold_reports"])


def test_oof_aggregate_rejects_length_mismatch_and_nonfinite_scores():
    frame = pd.DataFrame({"target": [0, 1], "category": ["a", "a"]})
    for base, candidate, folds in (
        ([0.1], [0.2, 0.8], [0, 1]),
        ([0.1, 0.9], [0.2, np.nan], [0, 1]),
    ):
        try:
            aggregate_oof_scores(frame, base, candidate, folds)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid OOF inputs must fail")
