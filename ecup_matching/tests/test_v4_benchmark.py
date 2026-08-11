from __future__ import annotations

import pandas as pd

from ecup_matching.ml.benchmark_v4_cuda import sample_benchmark_pairs


def test_v4_benchmark_sampler_is_exact_deterministic_and_keeps_both_classes() -> None:
    rows = []
    for category_index, category in enumerate(("A", "B", "C")):
        for target in (0, 1):
            for index in range(20):
                rows.append(
                    {
                        "id1": category_index * 1000 + target * 100 + index * 2,
                        "id2": category_index * 1000 + target * 100 + index * 2 + 1,
                        "target": target,
                        "category": category,
                    }
                )
    frame = pd.DataFrame(rows)

    first = sample_benchmark_pairs(frame, max_rows=30, seed=2026)
    second = sample_benchmark_pairs(frame, max_rows=30, seed=2026)

    assert len(first) == 30
    assert first[["id1", "id2"]].equals(second[["id1", "id2"]])
    assert set(first["category"]) == {"A", "B", "C"}
    assert set(first["target"]) == {0, 1}
    counts = first.groupby(["category", "target"]).size()
    assert counts.min() >= 1


def test_v4_benchmark_sampler_does_not_require_preserving_all_positives() -> None:
    frame = pd.DataFrame(
        {
            "id1": range(200),
            "id2": range(1000, 1200),
            "target": [1] * 150 + [0] * 50,
            "category": ["A"] * 200,
        }
    )

    sampled = sample_benchmark_pairs(frame, max_rows=32, seed=2026)

    assert len(sampled) == 32
    assert int((sampled["target"] == 1).sum()) < 150
    assert set(sampled["target"]) == {0, 1}
