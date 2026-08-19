"""Build one immutable proxy fixture from independently admitted never-labelled pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd


def _key(seed: int, row) -> str:
    return hashlib.sha256(f"{seed}\0{row.category}\0{int(row.target)}\0{row.reason_code}\0{row.id1}\0{row.id2}".encode()).hexdigest()


def _balanced(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    work = frame.copy().reset_index(drop=True)
    work["_balance"] = work["category"].astype(str) + "|" + work["target"].astype(int).astype(str) + "|" + work["reason_code"].astype(str)
    work["_key"] = [_key(seed, r) for r in work.itertuples(index=False)]
    if len(work) <= max_rows:
        return work.sort_values("_key", kind="mergesort").drop(columns=["_balance", "_key"]).reset_index(drop=True)
    groups = list(work.groupby("_balance", sort=True))
    quota = max(1, max_rows // max(1, len(groups)))
    picked = []
    used: set[int] = set()
    for _, group in groups:
        take = group.sort_values("_key", kind="mergesort").head(min(quota, len(group)))
        picked.append(take); used.update(take.index.tolist())
    out = pd.concat(picked, axis=0) if picked else work.iloc[:0]
    if len(out) < max_rows:
        pool = work.loc[~work.index.isin(used)].sort_values("_key", kind="mergesort")
        out = pd.concat([out, pool.head(max_rows - len(out))], axis=0)
    return out.sort_values("_key", kind="mergesort").head(max_rows).drop(columns=["_balance", "_key"]).reset_index(drop=True)


def build_proxy_fixture(
    *, labels_path: Path, item_db: Path, output_dir: Path, max_rows: int,
    seed: int, training_forbidden_ids: Iterable[int] = (),
) -> dict[str, object]:
    labels = pd.read_parquet(labels_path).reset_index(drop=True)
    required = {"id1", "id2", "target", "category", "reason_code"}
    if not required.issubset(labels.columns):
        raise ValueError(f"proxy labels missing columns: {sorted(required - set(labels.columns))}")
    forbidden = set(map(int, training_forbidden_ids))
    overlap_mask = labels["id1"].astype(int).isin(forbidden) | labels["id2"].astype(int).isin(forbidden)
    if overlap_mask.any():
        raise ValueError(f"proxy labels overlap training universe on {int(overlap_mask.sum())} rows")
    labels = _balanced(labels, int(max_rows), int(seed))
    if labels.empty:
        raise ValueError("proxy fixture cannot be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_cols = [c for c in ["id1", "id2", "target", "category", "reason_code", "stratum", "stratum_reliability"] if c in labels]
    truth = labels[truth_cols].copy().reset_index(drop=True)
    truth.to_parquet(output_dir / "proxy-truth.parquet", index=False)
    truth[["id1", "id2"]].to_parquet(output_dir / "proxy-matches.parquet", index=False)

    needed = sorted(set(truth.id1.astype(int)) | set(truth.id2.astype(int)))
    conn = sqlite3.connect(f"file:{item_db}?mode=ro", uri=True)
    rows = []
    for start in range(0, len(needed), 900):
        chunk = needed[start:start + 900]
        marks = ",".join("?" for _ in chunk)
        rows.extend(conn.execute(
            f"SELECT id,name,attributes,category FROM item WHERE id IN ({marks})", chunk
        ).fetchall())
    conn.close()
    items = pd.DataFrame(rows, columns=["id", "name", "attributes", "category"])
    if set(items.id.astype(int)) != set(needed):
        raise RuntimeError("proxy item DB did not contain every endpoint")
    items.sort_values("id", kind="mergesort").to_parquet(output_dir / "proxy-items.parquet", index=False)
    report = {
        "version": "v20-never-labelled-proxy-v1", "rows": int(len(truth)),
        "items": int(len(needed)), "categories": int(truth.category.astype(str).nunique()),
        "positive_rate": float((truth.target.astype(float) >= 0.5).mean()),
        "item_overlap_with_training_universe": 0, "max_rows": int(max_rows), "seed": int(seed),
        "sealed_gold_opened": False,
    }
    (output_dir / "proxy-manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--item-db", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--training-forbidden-ids", type=Path)
    a = p.parse_args()
    forbidden = []
    if a.training_forbidden_ids and a.training_forbidden_ids.exists():
        f = pd.read_parquet(a.training_forbidden_ids) if a.training_forbidden_ids.suffix == ".parquet" else pd.read_csv(a.training_forbidden_ids)
        col = "id" if "id" in f else f.columns[0]
        forbidden = f[col].astype(int).tolist()
    build_proxy_fixture(labels_path=a.labels, item_db=a.item_db, output_dir=a.output_dir,
                        max_rows=a.max_rows, seed=a.seed, training_forbidden_ids=forbidden)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
