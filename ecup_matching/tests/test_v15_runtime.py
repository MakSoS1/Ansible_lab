import pandas as pd

from ecup_matching.v15_runtime import materialize_referenced_items, write_predictions


def test_v15_runtime_materializes_only_referenced_items():
    items = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "attributes": ["{}"] * 4, "category": ["x"] * 4})
    pairs = pd.DataFrame({"id1": [1, 1], "id2": [2, 3]})
    subset = materialize_referenced_items(items, pairs)
    assert list(subset["id"]) == [1, 2, 3]


def test_v15_runtime_writes_exact_ordered_submission_columns(tmp_path):
    pairs = pd.DataFrame({"id1": [7, 1, 9], "id2": [8, 2, 10]})
    path = tmp_path / "submit.csv"
    write_predictions(pairs, [0.9, 0.1, 0.55], path)
    out = pd.read_csv(path)
    assert list(out.columns) == ["id1", "id2", "predict"]
    assert list(zip(out.id1, out.id2)) == [(7, 8), (1, 2), (9, 10)]
    assert out["predict"].notna().all()
