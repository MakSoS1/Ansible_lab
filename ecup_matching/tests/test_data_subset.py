import pandas as pd

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
