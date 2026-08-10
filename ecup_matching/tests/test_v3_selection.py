import numpy as np
import pandas as pd

from ecup_matching.ml.v3_selection import select_best_blend, select_hard_negatives


def test_select_hard_negatives_uses_only_negatives_and_prefers_high_scores():
    frame = pd.DataFrame(
        {
            "id1": list(range(8)),
            "id2": list(range(100, 108)),
            "target": [0, 0, 0, 0, 0, 1, 1, 1],
            "category": ["Электроника", "Электроника", "Аптека", "Аптека", "Аптека", "Электроника", "Аптека", "Аптека"],
        }
    )
    scores = np.array([0.99, 0.70, 0.98, 0.80, 0.20, 1.0, 1.0, 1.0])

    picked = select_hard_negatives(
        frame,
        scores,
        count=3,
        priority_categories={"Электроника"},
        priority_fraction=0.50,
        seed=2026,
    )

    assert len(picked) == 3
    assert (picked["target"].astype(float) < 0.5).all()
    # ceil(3 * .5) gives two reserved priority negatives when available.
    assert int((picked["category"] == "Электроника").sum()) == 2
    assert set(picked.loc[picked["category"] == "Электроника", "id1"]) == {0, 1}
    assert 2 in set(picked["id1"])  # highest remaining global false positive


def test_select_hard_negatives_is_deterministic_for_ties():
    frame = pd.DataFrame(
        {
            "id1": [1, 2, 3, 4],
            "id2": [11, 12, 13, 14],
            "target": [0, 0, 0, 0],
            "category": ["A", "A", "B", "B"],
        }
    )
    scores = np.array([0.8, 0.8, 0.8, 0.8])
    kwargs = dict(count=2, priority_categories={"A"}, priority_fraction=0.5, seed=9)

    first = select_hard_negatives(frame, scores, **kwargs)
    second = select_hard_negatives(frame, scores, **kwargs)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_select_best_blend_finds_improving_global_alpha_and_preserves_range():
    # Neural fixes the ranking inside B while structured already ranks A well.
    target = np.array([1, 0, 1, 0], dtype=float)
    category = np.array(["A", "A", "B", "B"], dtype=object)
    structured = np.array([0.9, 0.1, 0.2, 0.8], dtype=float)
    neural = np.array([0.8, 0.2, 0.9, 0.1], dtype=float)

    result = select_best_blend(
        structured,
        neural,
        target,
        category,
        alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
    )

    assert result["macro_average_precision"] >= result["structured_macro_average_precision"]
    assert 0.0 <= result["alpha_neural"] <= 1.0
    scores = np.asarray(result["scores"], dtype=float)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_select_best_blend_rejects_misaligned_lengths():
    target = np.array([1, 0], dtype=float)
    category = np.array(["A", "A"], dtype=object)

    try:
        select_best_blend(
            np.array([0.1]),
            np.array([0.2, 0.3]),
            target,
            category,
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("misaligned validation arrays must fail closed")
