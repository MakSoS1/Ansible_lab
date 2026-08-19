from __future__ import annotations

import pandas as pd

from ecup_matching.ml.run_v20_prepare_teacher_bakeoff import build_bakeoff_slice


def _frame() -> pd.DataFrame:
    rows = []
    idx = 0
    for category in ("phones", "laptops", "shoes"):
        for reason, target in (("SAME_MODEL", 1), ("MODEL_CONFLICT", 0), ("CAPACITY_CONFLICT", 0)):
            for _ in range(200):
                rows.append(
                    {
                        "id1": idx,
                        "id2": 100000 + idx,
                        "target": target,
                        "category": category,
                        "stratum": f"{category}|{reason}|hard",
                        "reason_code": reason,
                        "difficulty": "hard",
                    }
                )
                idx += 1
    return pd.DataFrame(rows)


def test_bakeoff_slice_is_deterministic_and_bounded():
    frame = _frame()
    pairs1, truth1, report1 = build_bakeoff_slice(frame, max_rows=500, seed=2026)
    pairs2, truth2, report2 = build_bakeoff_slice(frame, max_rows=500, seed=2026)

    assert pairs1.equals(pairs2)
    assert truth1.equals(truth2)
    assert report1 == report2
    assert len(pairs1) == 500
    assert len(truth1) == 500


def test_teacher_pairs_are_target_blind_but_truth_keeps_target():
    pairs, truth, _ = build_bakeoff_slice(_frame(), max_rows=400, seed=2026)
    assert "target" not in pairs.columns
    assert "target" in truth.columns
    assert set(pairs[["id1", "id2"]].itertuples(index=False, name=None)) == set(
        truth[["id1", "id2"]].itertuples(index=False, name=None)
    )


def test_slice_preserves_label_reason_and_category_coverage():
    _, truth, report = build_bakeoff_slice(_frame(), max_rows=500, seed=2026)
    assert set(truth["target"].astype(int)) == {0, 1}
    assert set(truth["reason_code"].astype(str)) == {"SAME_MODEL", "MODEL_CONFLICT", "CAPACITY_CONFLICT"}
    assert set(truth["category"].astype(str)) == {"phones", "laptops", "shoes"}
    assert report["groups_selected"] == 9
