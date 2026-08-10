import pandas as pd
import pyarrow as pa

from ecup_matching.ml import data_subset
from ecup_matching.ml.data_subset import select_items_by_ids


def test_select_items_by_ids_streams_parquet_and_preserves_requested_rows(tmp_path):
    path = tmp_path / "items.parquet"
    pd.DataFrame(
        {
            "id": [10, 11, 12, 13],
            "name": ["a", "b", "c", "d"],
            "attributes": ["{}"] * 4,
            "category": ["x", "x", "y", "y"],
            "unused": [1, 2, 3, 4],
        }
    ).to_parquet(path, index=False)
    out = select_items_by_ids(path, {13, 10}, batch_size=2)
    assert list(out.columns) == ["id", "name", "attributes", "category"]
    assert set(out["id"]) == {10, 13}


def test_select_items_by_ids_reports_missing(tmp_path):
    path = tmp_path / "items.parquet"
    pd.DataFrame({"id": [1], "name": ["a"], "attributes": ["{}"], "category": ["x"]}).to_parquet(path, index=False)
    try:
        select_items_by_ids(path, {1, 2})
        assert False, "expected missing ID failure"
    except KeyError as exc:
        assert "missing" in str(exc).lower()


def test_select_items_by_ids_filters_arrow_before_materializing_heavy_columns(
    monkeypatch, tmp_path
):
    batch = pa.RecordBatch.from_pydict(
        {
            "id": [10, 11, 12],
            "name": ["wanted", "unused-a", "unused-b"],
            "attributes": ["{}", "x" * 1000, "y" * 1000],
            "category": ["x", "x", "y"],
        }
    )
    observed = {}

    class GuardedBatch:
        def __init__(self, inner, *, filtered=False):
            self.inner = inner
            self.filtered = filtered
            self.schema = inner.schema
            self.num_rows = inner.num_rows

        def column(self, index):
            return self.inner.column(index)

        def filter(self, mask):
            return GuardedBatch(self.inner.filter(mask), filtered=True)

        def to_pandas(self):
            if not self.filtered:
                raise AssertionError("heavy columns were materialized before ID filtering")
            return self.inner.to_pandas()

    class FakeParquet:
        schema_arrow = batch.schema

        def iter_batches(self, **kwargs):
            observed["batch_size"] = kwargs["batch_size"]
            yield GuardedBatch(batch)

    class FakeMemoryPool:
        def release_unused(self):
            observed["release_calls"] = observed.get("release_calls", 0) + 1

    monkeypatch.setattr(data_subset.pq, "ParquetFile", lambda _path: FakeParquet())
    monkeypatch.setattr(
        data_subset.pa, "default_memory_pool", lambda: FakeMemoryPool()
    )

    out = select_items_by_ids(tmp_path / "unused.parquet", {10})

    assert out.to_dict("records") == [
        {"id": 10, "name": "wanted", "attributes": "{}", "category": "x"}
    ]
    assert observed["batch_size"] <= 5_000
    assert observed["release_calls"] >= 1
