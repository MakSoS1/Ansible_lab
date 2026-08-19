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


def test_select_items_by_ids_can_skip_heavy_attributes_for_name_only_smoke(tmp_path):
    path = tmp_path / "items.parquet"
    pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["wanted", "unused"],
            "attributes": ["x" * 1000, "y" * 1000],
            "category": ["x", "y"],
        }
    ).to_parquet(path, index=False)

    out = select_items_by_ids(path, {1}, include_attributes=False)

    assert out.to_dict("records") == [
        {"id": 1, "name": "wanted", "attributes": "{}", "category": "x"}
    ]


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
        """Fails loudly if unfiltered heavy columns are ever converted to pandas."""

        def __init__(self, inner):
            self.inner = inner
            self.schema = inner.schema
            self.num_rows = inner.num_rows

        def column(self, index):
            return self.inner.column(index)

        def filter(self, mask):
            return self.inner.filter(mask)

        def to_pandas(self):
            raise AssertionError("heavy columns were materialized before ID filtering")

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
    assert observed["release_calls"] >= 1


def test_select_items_by_ids_builds_the_requested_value_set_once(monkeypatch, tmp_path):
    """Rebuilding the value set per batch made the scan quadratic in requested ids."""
    rows = 40
    per_batch = 4
    batches = [
        pa.RecordBatch.from_pydict(
            {
                "id": list(range(start, start + per_batch)),
                "name": [f"n{i}" for i in range(start, start + per_batch)],
                "attributes": ["{}"] * per_batch,
                "category": ["x"] * per_batch,
            }
        )
        for start in range(0, rows, per_batch)
    ]

    class FakeParquet:
        schema_arrow = batches[0].schema

        def iter_batches(self, **kwargs):
            yield from batches

    real_array = data_subset.pa.array
    calls = {"count": 0}

    def counting_array(*args, **kwargs):
        calls["count"] += 1
        return real_array(*args, **kwargs)

    monkeypatch.setattr(data_subset.pq, "ParquetFile", lambda _path: FakeParquet())
    monkeypatch.setattr(data_subset.pa, "array", counting_array)

    out = select_items_by_ids(tmp_path / "unused.parquet", set(range(rows)))

    assert len(out) == rows
    assert calls["count"] == 1, (
        f"value set must be built once, not once per batch (built {calls['count']} times)"
    )


def test_select_items_by_ids_handles_duplicate_ids_without_early_exit(tmp_path):
    path = tmp_path / "items.parquet"
    pd.DataFrame(
        {
            "id": [1, 1, 1, 2],
            "name": ["first", "dup", "dup", "second"],
            "attributes": ["{}"] * 4,
            "category": ["x", "x", "x", "y"],
        }
    ).to_parquet(path, index=False)

    out = select_items_by_ids(path, {1, 2}, batch_size=1)

    assert out.to_dict("records") == [
        {"id": 1, "name": "first", "attributes": "{}", "category": "x"},
        {"id": 2, "name": "second", "attributes": "{}", "category": "y"},
    ]
