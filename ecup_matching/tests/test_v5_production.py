import numpy as np
import pandas as pd

from ecup_matching.ml.v5_fixed_blend import percentile_rank
from ecup_matching.ml.v5_production import (
    category_shrunk_hgb_equal_rank_fusion,
    category_shrunk_six_rank_fusion,
    final_six_rank_fusion,
    select_full_contrastive_pairs,
)


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


def _signals():
    return {
        "weak": [0.9, 0.1, 0.4, 0.2],
        "sparse": [9.0, 1.0, 4.0, 2.0],
        "explicit": [0.8, 0.2, 0.6, 0.1],
        "contrastive": [0.7, 0.3, 0.5, 0.2],
        "teacher": [0.95, 0.05, 0.7, 0.1],
        "typed_explicit": [0.85, 0.15, 0.55, 0.25],
    }


def _category_model():
    return {
        "signal_names": [
            "weak",
            "sparse",
            "explicit",
            "contrastive",
            "teacher",
            "typed_explicit",
        ],
        "category_weights": {
            "a": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        },
    }


class _FakeHGB:
    def predict_proba(self, design):
        x = np.asarray(design, dtype=np.float64)
        raw = 0.65 * x[:, 2] + 0.25 * x[:, 5] + 0.10 * (x[:, 6] / 2.0)
        raw = np.clip(raw, 0.0, 1.0)
        return np.column_stack([1.0 - raw, raw])


def test_final_six_rank_fusion_is_target_free_and_bounded():
    score = final_six_rank_fusion(_signals())
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


def test_category_shrunk_rank_fusion_applies_category_specific_simplex():
    categories = np.asarray(["a", "b", "a", "b"])
    score = category_shrunk_six_rank_fusion(_signals(), categories, _category_model())

    weak_rank = np.asarray([1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0])
    sparse_rank = np.asarray([1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0])
    expected = np.asarray([weak_rank[0], sparse_rank[1], weak_rank[2], sparse_rank[3]])
    assert np.allclose(score, expected)
    assert ((score >= 0.0) & (score <= 1.0)).all()


def test_category_shrunk_rank_fusion_rejects_unknown_category():
    model = {
        "signal_names": [
            "weak",
            "sparse",
            "explicit",
            "contrastive",
            "teacher",
            "typed_explicit",
        ],
        "category_weights": {"a": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    }
    try:
        category_shrunk_six_rank_fusion(_signals(), ["a", "a", "missing", "a"], model)
    except ValueError as exc:
        assert "category" in str(exc).lower()
    else:
        raise AssertionError("unknown production category must fail closed")


def test_category_shrunk_rank_fusion_rejects_invalid_weights():
    model = {
        "signal_names": [
            "weak",
            "sparse",
            "explicit",
            "contrastive",
            "teacher",
            "typed_explicit",
        ],
        "category_weights": {"a": [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]},
    }
    try:
        category_shrunk_six_rank_fusion(_signals(), ["a", "a", "a", "a"], model)
    except ValueError as exc:
        assert "weights" in str(exc).lower()
    else:
        raise AssertionError("non-simplex production weights must fail")


def test_category_hgb_equal_rank_fusion_matches_frozen_formula():
    signals = _signals()
    categories = np.asarray(["a", "b", "a", "b"])
    category_model = _category_model()
    hgb_bundle = {
        "signal_names": list(category_model["signal_names"]),
        "category_names": ["a", "b"],
        "model": _FakeHGB(),
    }

    actual = category_shrunk_hgb_equal_rank_fusion(
        signals,
        categories,
        category_model,
        hgb_bundle,
    )

    category_score = category_shrunk_six_rank_fusion(signals, categories, category_model)
    ranks = np.column_stack([percentile_rank(signals[name]) for name in category_model["signal_names"]])
    category_codes = np.asarray([0.0, 1.0, 0.0, 1.0])
    design = np.column_stack([ranks, category_codes])
    hgb_score = _FakeHGB().predict_proba(design)[:, 1]
    expected = 0.5 * percentile_rank(category_score) + 0.5 * percentile_rank(hgb_score)

    assert np.allclose(actual, expected)
    assert np.isfinite(actual).all()
    assert ((actual >= 0.0) & (actual <= 1.0)).all()


def test_category_hgb_equal_rank_fusion_rejects_unknown_hgb_category():
    hgb_bundle = {
        "signal_names": list(_category_model()["signal_names"]),
        "category_names": ["a"],
        "model": _FakeHGB(),
    }
    try:
        category_shrunk_hgb_equal_rank_fusion(
            _signals(),
            ["a", "b", "a", "b"],
            _category_model(),
            hgb_bundle,
        )
    except ValueError as exc:
        assert "category" in str(exc).lower()
    else:
        raise AssertionError("unknown HGB production category must fail closed")


def test_category_hgb_equal_rank_fusion_rejects_signal_order_mismatch():
    hgb_bundle = {
        "signal_names": [
            "sparse",
            "weak",
            "explicit",
            "contrastive",
            "teacher",
            "typed_explicit",
        ],
        "category_names": ["a", "b"],
        "model": _FakeHGB(),
    }
    try:
        category_shrunk_hgb_equal_rank_fusion(
            _signals(),
            ["a", "b", "a", "b"],
            _category_model(),
            hgb_bundle,
        )
    except ValueError as exc:
        assert "signal" in str(exc).lower()
    else:
        raise AssertionError("HGB signal order mismatch must fail closed")
