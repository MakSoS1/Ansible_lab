from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_oof_aggregate import aggregate_oof_scores
from .v5_validation import manifest_sha256


def aggregate_weighted_specialists(
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
    actual = manifest_sha256(manifest)
    if actual != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual}")
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, expected_folds = development_rows_and_folds(manifest, total_rows=len(matches))

    if len(fold_files) != len(np.unique(expected_folds)):
        raise ValueError("weighted specialist aggregate requires one file per development fold")
    pieces = [pd.read_parquet(path) for path in fold_files]
    candidate = pd.concat(pieces, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    if candidate["row_index"].duplicated().any():
        raise ValueError("duplicate weighted specialist OOF row")
    if candidate["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("weighted specialist OOF rows do not cover sealed development rows")
    if candidate["fold"].astype(np.int16).to_numpy().tolist() != expected_folds.tolist():
        raise ValueError("weighted specialist folds do not match sealed manifest")

    dev_pairs = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev_pairs["id1"], dev_pairs["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev_pairs["category"] = dev_pairs["id1"].map(category_by_id)
    if dev_pairs["category"].isna().any():
        raise RuntimeError("failed to attach development categories")

    base_frame = pd.read_parquet(category_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if base_frame["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("retained category base does not align with sealed development rows")
    result = aggregate_oof_scores(
        dev_pairs,
        base_frame["score"].to_numpy(dtype=np.float64),
        candidate["score"].to_numpy(dtype=np.float64),
        expected_folds,
    )
    payload = {
        "version": "v5b-weighted-category-specialist",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev_pairs)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "category_base_oof_macro_ap": float(result["base_macro_average_precision"]),
        "weighted_specialist_oof_macro_ap": float(result["macro_average_precision"]),
        "delta_vs_category_base": float(result["delta_vs_base"]),
        "fold_reports": result["fold_reports"],
        "per_category_ap": result["per_category_ap"],
    }
    (output_dir / "v5b-weighted-specialist-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    candidate.to_parquet(output_dir / "v5b-weighted-specialist-oof.parquet", index=False)
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
    payload = aggregate_weighted_specialists(
        fold_files=sorted(args.fold_dir.rglob("v5b-weighted-fold-*-oof.parquet")),
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
