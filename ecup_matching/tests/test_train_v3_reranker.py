import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.train_v3_reranker import (
    _select_accelerator,
    category_aware_blend,
    make_stage2_frame,
)


class _Backend:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self):
        return self._available


class _Torch:
    def __init__(self, cuda: bool, mps: bool):
        self.cuda = _Backend(cuda)
        self.backends = type("Backends", (), {"mps": _Backend(mps)})()


def test_select_accelerator_prefers_cuda_then_mps_and_rejects_cpu():
    assert _select_accelerator(_Torch(cuda=True, mps=True)) == "cuda"
    assert _select_accelerator(_Torch(cuda=False, mps=True)) == "mps"
    with pytest.raises(RuntimeError, match="accelerator"):
        _select_accelerator(_Torch(cuda=False, mps=False))


def test_make_stage2_frame_uses_human_examples_only_and_priority_false_positives():
    train = pd.DataFrame(
        {
            "id1": [1, 2, 3, 4, 5, 6, 7, 8],
            "id2": [11, 12, 13, 14, 15, 16, 17, 18],
            "target": [0, 0, 0, 0, 1, 1, 0, 1],
            "category": ["Электроника", "Электроника", "Аптека", "Аптека", "Электроника", "Аптека", "Электроника", "Электроника"],
            "source": ["human", "human", "human", "human", "human", "human", "weak", "weak"],
            "sample_weight": [1.0] * 8,
            "text_a": [f"a{i}" for i in range(8)],
            "text_b": [f"b{i}" for i in range(8)],
        }
    )
    human = train[train["source"] == "human"].reset_index(drop=True)
    # Scores align to all human rows; first two priority negatives are hardest.
    scores = np.array([0.99, 0.91, 0.80, 0.70, 0.20, 0.10], dtype=float)
    stage2, report = make_stage2_frame(
        train,
        human_scores=scores,
        hard_negative_count=3,
        priority_categories={"Электроника"},
        priority_fraction=0.67,
        seed=2026,
    )

    assert (stage2["source"] == "human").all()
    negatives = stage2[stage2["target"].astype(float) < 0.5]
    assert set(negatives["id1"]) == {1, 2, 3}
    assert report["selected_negatives"] == 3
    assert report["selected_positives"] >= 1


def test_category_aware_blend_never_reduces_priority_category_validation_ap():
    frame = pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 1, 0, 1, 0],
            "category": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "structured": [0.9, 0.2, 0.8, 0.1, 0.2, 0.9, 0.3, 0.8],
            "neural": [0.8, 0.1, 0.9, 0.2, 0.9, 0.1, 0.8, 0.2],
        }
    )
    result = category_aware_blend(
        frame,
        structured_col="structured",
        neural_col="neural",
        allowed_categories={"B"},
        alphas=(0.0, 0.5, 1.0),
    )
    assert result["category_alphas"]["A"] == 0.0
    assert result["category_alphas"]["B"] > 0.0
    assert result["macro_average_precision"] >= result["structured_macro_average_precision"]
    scores = np.asarray(result["scores"])
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
