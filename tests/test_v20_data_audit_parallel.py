from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecup_matching.ml.run_v20_data_audit import audit_weak_corpus, build_item_index


def _write_two_groups(path: Path, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    writer = pq.ParquetWriter(path, table.schema)
    try:
        mid = max(1, len(frame) // 2)
        writer.write_table(table.slice(0, mid))
        writer.write_table(table.slice(mid))
    finally:
        writer.close()


def _stable(report: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in report.items() if k != "elapsed_seconds"}


def test_parallel_weak_audit_is_exactly_equivalent_to_single_worker(tmp_path: Path) -> None:
    items = pd.DataFrame([
        {"id": 1, "name": "Samsung Galaxy S24 128GB", "attributes": '{"brand":"Samsung","storage":"128 GB"}', "category": "phone"},
        {"id": 2, "name": "Samsung Galaxy S24 128GB black", "attributes": '{"brand":"Samsung","storage":"128 GB"}', "category": "phone"},
        {"id": 3, "name": "Samsung Galaxy S24 256GB", "attributes": '{"brand":"Samsung","storage":"256 GB"}', "category": "phone"},
        {"id": 4, "name": "Чехол Samsung Galaxy S24", "attributes": '{"brand":"Samsung"}', "category": "phone"},
        {"id": 5, "name": "Футболка мужская размер M", "attributes": '{"размер":"M"}', "category": "clothes"},
        {"id": 6, "name": "Футболка мужская размер L", "attributes": '{"размер":"L"}', "category": "clothes"},
    ])
    weak = pd.DataFrame([
        {"id1": 1, "id2": 2, "target": 0.97},
        {"id1": 1, "id2": 3, "target": 0.08},
        {"id1": 1, "id2": 4, "target": 0.03},
        {"id1": 5, "id2": 6, "target": 0.12},
    ])
    items_path = tmp_path / "items.parquet"
    weak_path = tmp_path / "weak.parquet"
    _write_two_groups(items_path, items)
    _write_two_groups(weak_path, weak)

    db1 = tmp_path / "single.sqlite"
    db2 = tmp_path / "parallel.sqlite"
    build_item_index(items_path, db1)
    build_item_index(items_path, db2)

    single = audit_weak_corpus(weak_path=weak_path, item_db=db1, output_dir=tmp_path / "single", workers=1)
    parallel = audit_weak_corpus(weak_path=weak_path, item_db=db2, output_dir=tmp_path / "parallel", workers=2)

    assert _stable(parallel) == _stable(single)
    assert parallel["weak_rows"] == len(weak)
    assert parallel["weak_items"] == 6
    assert parallel["missing_item_pairs"] == 0
