from __future__ import annotations

import math

import pandas as pd

from ecup_matching.ml.v18_weak_quality import (
    continuous_weak_weight,
    prepare_weak_pairs_v18,
    split_weak_curriculum,
)


def test_continuous_weight_endpoints_dead_zone_and_symmetry() -> None:
    assert continuous_weak_weight(0.0) == 1.0
    assert continuous_weak_weight(1.0) == 1.0
    assert continuous_weak_weight(0.5) == 0.0
    assert continuous_weak_weight(0.55) == 0.0
    assert math.isclose(continuous_weak_weight(0.2), continuous_weak_weight(0.8), rel_tol=0, abs_tol=1e-12)
    assert 0.0 < continuous_weak_weight(0.65) < continuous_weak_weight(0.8) < 1.0


def test_prepare_retains_medium_confidence_and_deduplicates_by_margin() -> None:
    raw = pd.DataFrame(
        {
            "id1": [2, 1, 4, 5, 7],
            "id2": [1, 2, 3, 6, 8],
            "target": [0.80, 0.98, 0.65, 0.52, 0.10],
            "category": ["a", "a", "b", "c", "d"],
        }
    )
    out, report = prepare_weak_pairs_v18(raw)
    # (2,1) and (1,2) collapse; 0.98 is more confident than 0.80.
    pair = out[(out["id1"] == 1) & (out["id2"] == 2)].iloc[0]
    assert math.isclose(float(pair.target), 0.98)
    # 0.65 was discarded by the historical 0.30-0.70 dead band; v18 keeps it at low weight.
    medium = out[(out["id1"] == 3) & (out["id2"] == 4)].iloc[0]
    assert math.isclose(float(medium.target), 0.65)
    assert 0.0 < float(medium.weak_weight) < 0.3
    # 0.52 is inside the new narrow dead zone.
    assert not (((out["id1"] == 5) & (out["id2"] == 6)).any())
    assert report["input_rows"] == 5
    assert report["dead_zone_removed"] == 1
    assert report["duplicate_rows_removed"] == 1


def test_curriculum_high_is_subset_of_broad() -> None:
    raw = pd.DataFrame(
        {
            "id1": [1, 3, 5, 7],
            "id2": [2, 4, 6, 8],
            "target": [0.99, 0.85, 0.68, 0.12],
            "category": ["a", "a", "b", "b"],
        }
    )
    prepared, _ = prepare_weak_pairs_v18(raw)
    high, broad, report = split_weak_curriculum(prepared, high_margin=0.30)
    high_pairs = set(high[["id1", "id2"]].itertuples(index=False, name=None))
    broad_pairs = set(broad[["id1", "id2"]].itertuples(index=False, name=None))
    assert high_pairs <= broad_pairs
    assert report["broad_rows"] == len(broad)
    assert report["high_rows"] == len(high)
    assert len(high) < len(broad)
