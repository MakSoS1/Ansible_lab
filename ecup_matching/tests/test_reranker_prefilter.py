import numpy as np
import pandas as pd
import pyarrow as pa

from ecup_matching.ml import train_reranker_v2
from ecup_matching.ml.train_reranker_v2 import (
    _load_pair_items,
    _prefilter_weak,
    _prefilter_weak_parquet,
)


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


def test_streaming_parquet_prefilter_preserves_the_existing_sample_exactly(tmp_path) -> None:
    weak = pd.DataFrame(
        {
            "id1": np.arange(1_000),
            "id2": np.arange(20_000, 21_000),
            "target": np.resize([0.01, 0.10, 0.20, 0.50, 0.75, 0.90, 0.99], 1_000),
        }
    )
    path = tmp_path / "weak.parquet"
    weak.to_parquet(path, index=False, row_group_size=137)
    validation = {3, 18, 20_030, 20_111}

    expected = _legacy_prefilter(weak, validation, 125, 2026)
    actual, input_rows = _prefilter_weak_parquet(
        path,
        validation,
        125,
        2026,
        batch_size=83,
    )

    assert input_rows == len(weak)
    pd.testing.assert_frame_equal(actual, expected)


def test_streaming_prefilter_converts_only_selected_rows_to_pandas(monkeypatch, tmp_path):
    batch = pa.RecordBatch.from_pydict(
        {
            "id1": [1, 2, 3],
            "id2": [11, 12, 13],
            "target": [0.99, 0.50, 0.01],
        }
    )

    class GuardedBatch:
        def __init__(self, inner, *, selected=False):
            self.inner = inner
            self.selected = selected
            self.schema = inner.schema
            self.num_rows = inner.num_rows

        def column(self, index):
            return self.inner.column(index)

        def take(self, indices):
            return GuardedBatch(self.inner.take(indices), selected=True)

        def to_pandas(self):
            if not self.selected:
                raise AssertionError("weak batch was materialized before Arrow filtering")
            return self.inner.to_pandas()

    class FakeMetadata:
        num_rows = batch.num_rows

    class FakeParquet:
        schema_arrow = batch.schema
        metadata = FakeMetadata()

        def iter_batches(self, **_kwargs):
            yield GuardedBatch(batch)

    monkeypatch.setattr(
        train_reranker_v2.pq, "ParquetFile", lambda _path: FakeParquet()
    )

    out, input_rows = _prefilter_weak_parquet(
        tmp_path / "unused.parquet", set(), 2, 2026
    )

    assert input_rows == 3
    assert out.to_dict("records") == [
        {"id1": 1, "id2": 11, "target": 0.99},
        {"id1": 3, "id2": 13, "target": 0.01},
    ]


def test_load_pair_items_excludes_unreferenced_human_catalog_rows(tmp_path):
    path = tmp_path / "items_human.parquet"
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "unused"],
            "attributes": ["{}", "{}", "x" * 1000],
            "category": ["x", "x", "y"],
        }
    ).to_parquet(path, index=False)
    pairs = pd.DataFrame({"id1": [1], "id2": [2], "target": [1]})

    out = _load_pair_items(path, pairs)

    assert out["id"].tolist() == [1, 2]
