"""D4 candidate generation over real items using a disk-backed block index."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd

from .textnorm import normalize_item
from .v20_candidates import _block_signatures, _canonical_pair
from .v20_strata import classify_pair_stratum


GENERATOR_VERSION = "v20-disk-blocks-v1"


def _priority(seed: int, category: str, signature: str, a: int, b: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{category}\0{signature}\0{a}\0{b}".encode()).digest()


def build_block_index(item_db: Path, *, max_signatures_per_item: int = 8) -> dict[str, int]:
    conn = sqlite3.connect(item_db)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("DROP TABLE IF EXISTS block")
        conn.execute("CREATE TABLE block(category TEXT NOT NULL, signature TEXT NOT NULL, item_id INTEGER NOT NULL)")
        cursor = conn.execute("SELECT id,name,attributes,category FROM item ORDER BY id")
        rows = blocks = 0
        batch = []
        for item_id, name, attrs, category in cursor:
            norm = normalize_item(item_id, name, attrs, category)
            sigs = _block_signatures(norm)[:max_signatures_per_item]
            batch.extend((norm.category, sig, int(item_id)) for sig in sigs)
            rows += 1; blocks += len(sigs)
            if len(batch) >= 100_000:
                conn.executemany("INSERT INTO block(category,signature,item_id) VALUES(?,?,?)", batch)
                conn.commit(); batch.clear()
        if batch:
            conn.executemany("INSERT INTO block(category,signature,item_id) VALUES(?,?,?)", batch)
            conn.commit()
        conn.execute("CREATE INDEX block_key_idx ON block(category,signature,item_id)")
        conn.commit()
        return {"items": rows, "block_rows": blocks}
    finally:
        conn.close()


def generate_from_block_index(
    *,
    item_db: Path,
    output_path: Path,
    forbidden_ids: set[int],
    max_pairs: int,
    max_degree: int,
    max_pairs_per_category_reason: int,
    max_block_size: int,
    seed: int,
    exclude_weak_items: bool = False,
) -> dict[str, object]:
    conn = sqlite3.connect(f"file:{item_db}?mode=ro", uri=True)
    degree: Counter[int] = Counter()
    bucket: Counter[tuple[str, str]] = Counter()
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, object]] = []

    @lru_cache(maxsize=250_000)
    def item(item_id: int):
        hit = conn.execute("SELECT name,attributes,category FROM item WHERE id=?", (int(item_id),)).fetchone()
        if hit is None:
            raise KeyError(item_id)
        return normalize_item(item_id, hit[0], hit[1], hit[2])

    groups = conn.execute(
        "SELECT category,signature,COUNT(*) FROM block GROUP BY category,signature HAVING COUNT(*) BETWEEN 2 AND ? ORDER BY category,signature",
        (int(max_block_size),),
    )
    candidate_blocks = accepted_pairs = 0
    for category, signature, count in groups:
        candidate_blocks += 1
        if exclude_weak_items:
            ids = [int(r[0]) for r in conn.execute(
                "SELECT b.item_id FROM block b LEFT JOIN weak_item w ON w.id=b.item_id "
                "WHERE b.category=? AND b.signature=? AND w.id IS NULL ORDER BY b.item_id",
                (category, signature),
            )]
        else:
            ids = [int(r[0]) for r in conn.execute(
                "SELECT item_id FROM block WHERE category=? AND signature=? ORDER BY item_id",
                (category, signature),
            )]
        if len(ids) < 2:
            continue
        pairs: list[tuple[bytes, int, int]] = []
        for i, a in enumerate(ids):
            if a in forbidden_ids:
                continue
            for b in ids[i + 1:]:
                if b in forbidden_ids:
                    continue
                left, right = _canonical_pair(a, b)
                left, right = int(left), int(right)
                if (left, right) in seen:
                    continue
                pairs.append((_priority(seed, str(category), str(signature), left, right), left, right))
        for _, left, right in sorted(pairs):
            if len(rows) >= max_pairs:
                break
            if degree[left] >= max_degree or degree[right] >= max_degree:
                continue
            stratum = classify_pair_stratum(item(left), item(right))
            key = (stratum.category, stratum.reason_code)
            if bucket[key] >= max_pairs_per_category_reason:
                continue
            seen.add((left, right)); degree[left] += 1; degree[right] += 1; bucket[key] += 1
            rows.append({
                "id1": left, "id2": right, "category": stratum.category,
                "stratum": f"{stratum.category}|{stratum.reason_code}|{stratum.difficulty}",
                "reason_code": stratum.reason_code, "difficulty": stratum.difficulty,
                "generator_version": GENERATOR_VERSION, "generator_block": str(signature),
                "population": "never_labelled" if exclude_weak_items else "nonhuman_candidate",
            })
            accepted_pairs += 1
        if len(rows) >= max_pairs:
            break
    conn.close()
    columns = [
        "id1", "id2", "category", "stratum", "reason_code", "difficulty",
        "generator_version", "generator_block", "population",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return {
        "version": GENERATOR_VERSION,
        "pairs": int(len(frame)), "unique_items": int(len(set(frame.id1) | set(frame.id2))) if len(frame) else 0,
        "candidate_blocks": candidate_blocks, "accepted_pairs": accepted_pairs,
        "max_observed_degree": int(max(degree.values(), default=0)),
        "max_degree": int(max_degree), "max_pairs": int(max_pairs),
        "max_pairs_per_category_reason": int(max_pairs_per_category_reason),
        "forbidden_items": int(len(forbidden_ids)), "exclude_weak_items": bool(exclude_weak_items),
        "target_column_present": bool("target" in frame.columns),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--item-db", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--forbidden-ids", type=Path)
    p.add_argument("--max-pairs", type=int, default=1_000_000)
    p.add_argument("--max-degree", type=int, default=6)
    p.add_argument("--max-pairs-per-category-reason", type=int, default=25_000)
    p.add_argument("--max-block-size", type=int, default=40)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--rebuild-block-index", action="store_true")
    p.add_argument("--exclude-weak-items", action="store_true")
    args = p.parse_args()
    if args.rebuild_block_index:
        report = build_block_index(args.item_db)
        (args.output.parent / "block-index.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    forbidden: set[int] = set()
    if args.forbidden_ids and args.forbidden_ids.exists():
        data = pd.read_parquet(args.forbidden_ids) if args.forbidden_ids.suffix == ".parquet" else pd.read_csv(args.forbidden_ids)
        col = "id" if "id" in data else data.columns[0]
        forbidden = set(data[col].astype(int))
    report = generate_from_block_index(
        item_db=args.item_db, output_path=args.output, forbidden_ids=forbidden,
        max_pairs=args.max_pairs, max_degree=args.max_degree,
        max_pairs_per_category_reason=args.max_pairs_per_category_reason,
        max_block_size=args.max_block_size, seed=args.seed,
        exclude_weak_items=args.exclude_weak_items,
    )
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
