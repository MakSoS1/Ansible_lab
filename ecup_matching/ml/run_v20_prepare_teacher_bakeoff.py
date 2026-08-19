"""Build a deterministic, stratified, target-blind teacher bakeoff slice from fold-safe human audit rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _row_key(seed: int, id1: object, id2: object) -> bytes:
    return hashlib.sha256(f"{seed}\0{id1}\0{id2}".encode("utf-8")).digest()


def build_bakeoff_slice(
    audit_rows: pd.DataFrame,
    *,
    max_rows: int = 4_000,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = {"id1", "id2", "target", "category", "stratum", "reason_code", "difficulty"}
    if not required.issubset(audit_rows.columns):
        raise ValueError(f"audit rows missing columns: {sorted(required - set(audit_rows.columns))}")
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    work = audit_rows[list(required)].copy().reset_index(drop=True)
    if work.empty:
        raise ValueError("audit rows must not be empty")
    work["target"] = (work["target"].astype(float) >= 0.5).astype(int)
    work["_group"] = (
        work["target"].astype(str) + "\x1f"
        + work["reason_code"].astype(str) + "\x1f"
        + work["category"].astype(str)
    )

    queues: dict[str, list[int]] = {}
    for group_name, group in work.groupby("_group", sort=True):
        indices = list(group.index)
        indices.sort(key=lambda idx: _row_key(seed, work.at[idx, "id1"], work.at[idx, "id2"]))
        queues[str(group_name)] = indices

    selected: list[int] = []
    positions = {name: 0 for name in queues}
    group_names = sorted(queues)
    cap = min(int(max_rows), len(work))
    while len(selected) < cap:
        progressed = False
        for name in group_names:
            pos = positions[name]
            queue = queues[name]
            if pos >= len(queue):
                continue
            selected.append(queue[pos])
            positions[name] = pos + 1
            progressed = True
            if len(selected) >= cap:
                break
        if not progressed:
            break

    truth_columns = ["id1", "id2", "target", "category", "stratum", "reason_code", "difficulty"]
    truth = work.loc[selected, truth_columns].reset_index(drop=True)
    pair_columns = ["id1", "id2", "category", "stratum", "reason_code", "difficulty"]
    pairs = truth[pair_columns].copy()
    report = {
        "version": "v20-teacher-bakeoff-slice-v1",
        "source_rows": int(len(work)),
        "selected_rows": int(len(truth)),
        "max_rows": int(max_rows),
        "seed": int(seed),
        "groups_available": int(len(queues)),
        "groups_selected": int(truth.assign(_group=(
            truth["target"].astype(str) + "\x1f" + truth["reason_code"].astype(str) + "\x1f" + truth["category"].astype(str)
        ))["_group"].nunique()),
        "target_counts": {str(k): int(v) for k, v in truth["target"].value_counts().sort_index().items()},
        "reason_counts": {str(k): int(v) for k, v in truth["reason_code"].astype(str).value_counts().sort_index().items()},
        "category_counts": {str(k): int(v) for k, v in truth["category"].astype(str).value_counts().sort_index().items()},
        "teacher_target_column_present": bool("target" in pairs.columns),
        "sealed_gold_opened": False,
    }
    return pairs, truth, report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=4_000)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    audit = pd.read_parquet(a.audit)
    pairs, truth, report = build_bakeoff_slice(audit, max_rows=a.max_rows, seed=a.seed)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(a.output_dir / "teacher-bakeoff-pairs.parquet", index=False)
    truth.to_parquet(a.output_dir / "teacher-bakeoff-truth.parquet", index=False)
    (a.output_dir / "teacher-bakeoff.manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
