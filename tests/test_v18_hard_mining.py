from __future__ import annotations

import numpy as np
import pandas as pd

from ecup_matching.ml.v18_hard_mining import select_disagreement_hard_examples


def test_hard_mining_uses_disagreement_without_replacing_targets() -> None:
    frame = pd.DataFrame(
        {
            "id1": [1, 3, 5, 7, 9, 11],
            "id2": [2, 4, 6, 8, 10, 12],
            "target": [0.99, 0.95, 0.05, 0.10, 0.90, 0.08],
            "weak_weight": [1.0, 0.9, 1.0, 0.8, 0.8, 0.9],
            "hard_target": [1, 1, 0, 0, 1, 0],
            "category": ["a", "a", "a", "a", "b", "b"],
        }
    )
    predictions = np.array([0.10, 0.90, 0.90, 0.15, 0.20, 0.10], dtype=float)
    selected, report = select_disagreement_hard_examples(frame, predictions, max_rows=4, seed=2026)
    assert len(selected) == 4
    assert "_hard_disagreement" in selected.columns
    original_target = frame.set_index(["id1", "id2"])["target"].to_dict()
    for row in selected.itertuples(index=False):
        assert float(row.target) == float(original_target[(row.id1, row.id2)])
    # The two most obvious confident disagreements must survive balancing/fill.
    chosen = set(selected[["id1", "id2"]].itertuples(index=False, name=None))
    assert (1, 2) in chosen
    assert (5, 6) in chosen
    assert report["selected_rows"] == 4


def test_hard_mining_is_deterministic() -> None:
    n = 40
    frame = pd.DataFrame(
        {
            "id1": np.arange(0, 2 * n, 2),
            "id2": np.arange(1, 2 * n, 2),
            "target": np.where(np.arange(n) % 2 == 0, 0.9, 0.1),
            "weak_weight": np.linspace(0.2, 1.0, n),
            "hard_target": np.arange(n) % 2,
            "category": np.where(np.arange(n) % 3 == 0, "a", "b"),
        }
    )
    predictions = np.linspace(0.95, 0.05, n)
    a, _ = select_disagreement_hard_examples(frame, predictions, max_rows=15, seed=17)
    b, _ = select_disagreement_hard_examples(frame, predictions, max_rows=15, seed=17)
    assert a[["id1", "id2"]].equals(b[["id1", "id2"]])
