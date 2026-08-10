import numpy as np
import pandas as pd

from ecup_matching.ml.train_reranker_v2 import _prefilter_weak


def _legacy_prefilter(
    weak: pd.DataFrame,
    validation_item_ids: set[object],
    max_presample_rows: int,
    seed: int,
) -> pd.DataFrame:
    probability = pd.to_numeric(weak["target"], errors="raise").astype(float)
    weight = np.zeros(len(weak), dtype=np.float32)
    weight[(probability <= 0.03) | (probability >= 0.97)] = 1.0
    weight[
        ((probability > 0.03) & (probability <= 0.15))
        | ((probability >= 0.85) & (probability < 0.97))
    ] = 0.6
    weight[
        ((probability > 0.15) & (probability <= 0.30))
        | ((probability >= 0.70) & (probability < 0.85))
    ] = 0.3
    keep = weight > 0
    keep &= ~weak["id1"].isin(validation_item_ids).to_numpy()
    keep &= ~weak["id2"].isin(validation_item_ids).to_numpy()
    out = weak.loc[keep, ["id1", "id2", "target"]].copy().reset_index(drop=True)
    if len(out) > max_presample_rows:
        out = out.sample(n=max_presample_rows, random_state=seed).reset_index(drop=True)
    return out


def test_memory_bounded_prefilter_preserves_the_existing_sample_exactly() -> None:
    weak = pd.DataFrame(
        {
            "id1": np.arange(500),
            "id2": np.arange(10_000, 10_500),
            "target": np.resize([0.01, 0.10, 0.20, 0.50, 0.75, 0.90, 0.99], 500),
        }
    )
    validation = {3, 18, 10_030, 10_111}

    expected = _legacy_prefilter(weak, validation, 75, 2026)
    actual = _prefilter_weak(weak, validation, 75, 2026)

    pd.testing.assert_frame_equal(actual, expected)

