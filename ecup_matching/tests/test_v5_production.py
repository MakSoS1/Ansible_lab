import numpy as np
import pandas as pd

from ecup_matching.ml.v5_production import final_six_rank_fusion, select_full_contrastive_pairs


def test_full_contrastive_curriculum_keeps_all_positives_and_caps_negatives_deterministically():
    frame = pd.DataFrame(
        {
            "id1": np.arange(12),
            "id2": np.arange(100, 112),
            "target": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "category": ["a"] * 12,
        }
    )
    base = np.linspace(0.0, 1.0, len(frame))
    a = select_full_contrastive_pairs(frame, base, max_negative_to_positive=2.0, hard_negative_fraction=0.5, seed=2026)
    b = select_full_contrastive_pairs(frame, base, max_negative_to_positive=2.0, hard_negative_fraction=0.5, seed=2026)

    assert len(a) == 9
    assert int((a.target >= 0.5).sum()) == 3
    assert int((a.target < 0.5).sum()) == 6
    assert set(frame.loc[frame.target >= 0.5, "id1"]) <= set(a.id1)
    pd.testing.assert_frame_equal(a, b)


def test_final_six_rank_fusion_is_target_free_and_bounded():
    signals = {
        "weak": [0.9, 0.1, 0.4, 0.2],
        "sparse": [9.0, 1.0, 4.0, 2.0],
        "explicit": [0.8, 0.2, 0.6, 0.1],
        "contrastive": [0.7, 0.3, 0.5, 0.2],
        "teacher": [0.95, 0.05, 0.7, 0.1],
        "typed_explicit": [0.85, 0.15, 0.55, 0.25],
    }
    score = final_six_rank_fusion(signals)
    assert score.shape == (4,)
    assert np.isfinite(score).all()
    assert ((score >= 0.0) & (score <= 1.0)).all()
    assert score[0] == score.max()
    assert score[1] < score[2]


def test_final_six_rank_fusion_requires_exact_signal_set():
    signals = {name: [0.1, 0.2] for name in ("weak", "sparse", "explicit", "contrastive", "teacher")}
    try:
        final_six_rank_fusion(signals)
    except ValueError as exc:
        assert "signal" in str(exc).lower()
    else:
        raise AssertionError("missing typed explicit signal must fail")
