from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_evaluation import macro_ap_report
from .v5_semantic_stack import crossfit_semantic_stack
from .v5_validation import manifest_sha256


def aggregate_teacher_oof(
    *,
    fold_files: list[Path],
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    category_oof_path: Path,
    output_dir: Path,
    expected_split_sha: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_sha256(manifest) != expected_split_sha:
        raise ValueError("sealed split SHA mismatch")
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, expected_folds = development_rows_and_folds(manifest, total_rows=len(matches))
    if len(fold_files) != len(np.unique(expected_folds)):
        raise ValueError("teacher aggregate requires one file per development fold")
    teacher = pd.concat([pd.read_parquet(p) for p in fold_files], ignore_index=True)
    teacher = teacher.sort_values("row_index").reset_index(drop=True)
    if teacher["row_index"].duplicated().any():
        raise ValueError("duplicate teacher OOF row")
    if teacher["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("teacher OOF rows do not cover sealed development rows")
    if teacher["fold"].astype(np.int16).to_numpy().tolist() != expected_folds.tolist():
        raise ValueError("teacher OOF folds do not match sealed manifest")

    dev_pairs = matches.iloc[dev_rows].reset_index(drop=True)
    ids = pd.unique(pd.concat([dev_pairs["id1"], dev_pairs["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, ids, include_attributes=False)
    categories = items.set_index("id")["category"].astype(str)
    dev_pairs["category"] = dev_pairs["id1"].map(categories)
    if dev_pairs["category"].isna().any():
        raise RuntimeError("failed to attach categories")

    base = pd.read_parquet(category_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if base["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("category base rows do not align")
    base_scores = base["score"].to_numpy(dtype=np.float64)
    teacher_scores = teacher["teacher_score"].to_numpy(dtype=np.float64)
    direct = macro_ap_report(dev_pairs, teacher_scores)
    semantic = pd.DataFrame({"teacher_score": teacher_scores})
    stack = crossfit_semantic_stack(
        dev_pairs,
        base_scores,
        semantic,
        expected_folds,
        seed=2026,
        max_iter=220,
    )
    payload = {
        "version": "v5f-rubert-teacher-sprint",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev_pairs)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "category_base_oof_macro_ap": float(stack["base_macro_average_precision"]),
        "teacher_direct_oof_macro_ap": float(direct["macro_average_precision"]),
        "teacher_stack_oof_macro_ap": float(stack["macro_average_precision"]),
        "delta_vs_category_base": float(stack["delta_vs_base"]),
        "fold_reports": stack["fold_reports"],
        "per_category_ap": stack["per_category_ap"],
    }
    (output_dir / "v5f-teacher-sprint-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": expected_folds,
            "teacher_score": teacher_scores,
            "score": stack["scores"],
        }
    ).to_parquet(output_dir / "v5f-teacher-sprint-oof.parquet", index=False)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--category-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    args = parser.parse_args()
    payload = aggregate_teacher_oof(
        fold_files=sorted(args.fold_dir.rglob("v5f-teacher-fold-*-oof.parquet")),
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        category_oof_path=args.category_oof,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
