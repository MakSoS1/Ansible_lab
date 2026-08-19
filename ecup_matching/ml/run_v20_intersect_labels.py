"""Intersect independently admitted v20 label sets for production/proxy use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def run(first_path: Path, second_path: Path, output_path: Path) -> dict[str, object]:
    first = pd.read_parquet(first_path)
    second = pd.read_parquet(second_path)
    keys = ["id1", "id2"]
    merged = first.merge(second, on=keys, how="inner", suffixes=("_a", "_b"))
    if len(merged):
        disagreement = merged["target_a"].astype(int) != merged["target_b"].astype(int)
        if disagreement.any():
            raise RuntimeError(f"independent admission sets disagree on {int(disagreement.sum())} targets")
        rows = pd.DataFrame({
            "id1": merged.id1.astype(int), "id2": merged.id2.astype(int),
            "target": merged.target_a.astype(int),
            "category": merged.category_a.astype(str),
            "stratum": merged.stratum_a.astype(str),
            "reason_code": merged.reason_code_a.astype(str),
            "admitted": True,
            "stratum_reliability": merged[["stratum_reliability_a", "stratum_reliability_b"]].min(axis=1).astype(float),
            "label_origin": "two_teacher_two_policy_intersection",
            "teacher_ids": merged.teacher_ids_a.astype(str),
            "teacher_revisions": merged.teacher_revisions_a.astype(str),
            "prompt_sha256": merged.prompt_sha256_a.astype(str),
        })
    else:
        rows = pd.DataFrame(columns=[
            "id1", "id2", "target", "category", "stratum", "reason_code", "admitted",
            "stratum_reliability", "label_origin", "teacher_ids", "teacher_revisions", "prompt_sha256",
        ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output_path, index=False)
    report = {
        "version": "v20-two-policy-intersection-v1", "first_rows": int(len(first)),
        "second_rows": int(len(second)), "intersection_rows": int(len(rows)),
        "sealed_gold_opened": False,
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.first, a.second, a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
