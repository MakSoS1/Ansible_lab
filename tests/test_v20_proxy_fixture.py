from pathlib import Path
import sqlite3

import pandas as pd

from ecup_matching.ml.run_v20_proxy_fixture import build_proxy_fixture


def test_proxy_fixture_balances_and_materializes_exact_items(tmp_path: Path):
    labels = pd.DataFrame([
        {"id1": i * 2, "id2": i * 2 + 1, "target": i % 2, "category": "A" if i < 10 else "B", "reason_code": "OTHER", "stratum_reliability": 0.99}
        for i in range(20)
    ])
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path, index=False)
    db = tmp_path / "items.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE item(id INTEGER PRIMARY KEY,name TEXT,attributes TEXT,category TEXT)")
    ids = sorted(set(labels.id1) | set(labels.id2))
    conn.executemany("INSERT INTO item VALUES(?,?,?,?)", [(int(i), f"item {i}", "{}", "A" if i < 20 else "B") for i in ids])
    conn.commit(); conn.close()
    report = build_proxy_fixture(labels_path=labels_path, item_db=db, output_dir=tmp_path / "proxy", max_rows=12, seed=2026)
    truth = pd.read_parquet(tmp_path / "proxy" / "proxy-truth.parquet")
    items = pd.read_parquet(tmp_path / "proxy" / "proxy-items.parquet")
    matches = pd.read_parquet(tmp_path / "proxy" / "proxy-matches.parquet")
    assert len(truth) == len(matches) == 12
    assert set(items.id) == set(matches.id1) | set(matches.id2)
    assert set(matches.columns) == {"id1", "id2"}
    assert report["item_overlap_with_training_universe"] == 0
