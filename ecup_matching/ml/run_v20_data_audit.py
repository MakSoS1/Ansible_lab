"""D1: stream the complete weak corpus and build semantic strata without loading items.parquet into RAM."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time

import pandas as pd
import pyarrow.parquet as pq

from .textnorm import normalize_item
from .v20_policy import V20Policy, policy_sha256
from .v20_strata import classify_pair_stratum, target_band


def _emit(phase: str, **payload) -> None:
    print(json.dumps({"phase": phase, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def build_item_index(items_path: Path, db_path: Path) -> dict[str, object]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT NOT NULL, attributes TEXT NOT NULL, category TEXT NOT NULL)")
        conn.execute("CREATE TABLE weak_item (id INTEGER PRIMARY KEY)")
        handle = pq.ParquetFile(items_path)
        rows = 0
        started = time.perf_counter()
        for index in range(handle.metadata.num_row_groups):
            frame = handle.read_row_group(index, columns=["id", "name", "attributes", "category"]).to_pandas()
            records = [
                (int(r.id), "" if pd.isna(r.name) else str(r.name), "{}" if pd.isna(r.attributes) else str(r.attributes), "" if pd.isna(r.category) else str(r.category))
                for r in frame.itertuples(index=False)
            ]
            conn.executemany("INSERT INTO item(id,name,attributes,category) VALUES(?,?,?,?)", records)
            conn.commit()
            rows += len(records)
            _emit("v20-item-index", row_group=index + 1, row_groups=handle.metadata.num_row_groups, rows=rows)
        return {"rows": rows, "db_bytes": db_path.stat().st_size, "elapsed_seconds": time.perf_counter() - started}
    finally:
        conn.close()


def _fetch_items(conn: sqlite3.Connection, ids: list[int], *, query_chunk: int = 900) -> dict[int, object]:
    result: dict[int, object] = {}
    for start in range(0, len(ids), query_chunk):
        chunk = ids[start : start + query_chunk]
        marks = ",".join("?" for _ in chunk)
        for item_id, name, attrs, category in conn.execute(
            f"SELECT id,name,attributes,category FROM item WHERE id IN ({marks})", chunk
        ):
            result[int(item_id)] = normalize_item(int(item_id), name, attrs, category)
    return result


def audit_weak_corpus(*, weak_path: Path, item_db: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = pq.ParquetFile(weak_path)
    # Read item records and persist weak membership through separate connections;
    # WAL is unnecessary because this stage is single-process.
    read_conn = sqlite3.connect(f"file:{item_db}?mode=ro", uri=True)
    write_conn = sqlite3.connect(item_db)
    write_conn.execute("PRAGMA journal_mode=OFF")
    write_conn.execute("PRAGMA synchronous=OFF")
    counts: Counter[tuple[str, str, str, str, int]] = Counter()
    missing_items = pair_rows = 0
    target_sum = 0.0
    started = time.perf_counter()
    try:
        for index in range(handle.metadata.num_row_groups):
            frame = handle.read_row_group(index, columns=["id1", "id2", "target"]).to_pandas()
            ids = sorted(set(frame["id1"].astype(int)) | set(frame["id2"].astype(int)))
            write_conn.executemany("INSERT OR IGNORE INTO weak_item(id) VALUES(?)", ((item_id,) for item_id in ids))
            write_conn.commit()
            items = _fetch_items(read_conn, ids)
            for row in frame.itertuples(index=False):
                left = items.get(int(row.id1)); right = items.get(int(row.id2))
                if left is None or right is None:
                    missing_items += 1
                    continue
                s = classify_pair_stratum(left, right)
                p = float(row.target)
                counts[(s.category, s.reason_code, s.difficulty, target_band(p), int(p >= 0.5))] += 1
                pair_rows += 1
                target_sum += p
            _emit("v20-weak-audit", row_group=index + 1, row_groups=handle.metadata.num_row_groups, pairs=pair_rows, strata=len(counts))
    finally:
        read_conn.close(); write_conn.close()

    verify = sqlite3.connect(f"file:{item_db}?mode=ro", uri=True)
    weak_items = int(verify.execute("SELECT COUNT(*) FROM weak_item").fetchone()[0])
    total_items = int(verify.execute("SELECT COUNT(*) FROM item").fetchone()[0])
    verify.close()
    strata = [
        {"category": k[0], "reason_code": k[1], "difficulty": k[2], "target_band": k[3], "hard_target": k[4], "count": int(v)}
        for k, v in sorted(counts.items())
    ]
    policy = V20Policy()
    report = {
        "version": "v20-semantic-audit-v1",
        "weak_rows": int(pair_rows), "weak_items": weak_items, "total_items": total_items,
        "nonweak_items": int(total_items - weak_items),
        "missing_item_pairs": int(missing_items),
        "target_mean": float(target_sum / max(pair_rows, 1)),
        "strata_rows": int(len(strata)),
        "policy_sha256": policy_sha256(policy),
        "sealed_gold_opened": False,
        "gold_rows_scored": 0,
        "elapsed_seconds": float(time.perf_counter() - started),
        "strata": strata,
    }
    (output_dir / "STRATA.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--weak-matches", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--item-db", type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db = args.item_db or (args.output_dir / "items-v20.sqlite")
    index_report = build_item_index(args.items, db)
    (args.output_dir / "item-index.json").write_text(json.dumps(index_report, indent=2, sort_keys=True) + "\n")
    audit_weak_corpus(weak_path=args.weak_matches, item_db=db, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
