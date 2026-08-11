from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_attribute_stack import crossfit_attribute_evidence_stack
from .v5_validation import manifest_sha256


def run_attribute_oof(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    base_oof_path: Path,
    output_dir: Path,
    expected_split_sha: str,
    min_support: int = 20,
    smoothing: float = 2.0,
    evidence_clip: float = 8.0,
) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = manifest_sha256(manifest)
    if actual_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_sha}")

    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    dev_pairs = matches.iloc[dev_rows].reset_index(drop=True)
    dev_item_ids = pd.unique(pd.concat([dev_pairs["id1"], dev_pairs["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, dev_item_ids, include_attributes=True)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev_pairs["category"] = dev_pairs["id1"].map(category_by_id)
    if dev_pairs["category"].isna().any():
        raise RuntimeError("failed to attach development categories")

    gold_rows = np.asarray(manifest["gold_rows"], dtype=np.int64)
    gold_pairs = matches.iloc[gold_rows]
    gold_items = set(gold_pairs["id1"].tolist()) | set(gold_pairs["id2"].tolist())
    overlap = gold_items & set(items["id"].tolist())
    if overlap:
        raise RuntimeError(f"gold items leaked into attribute development set: {len(overlap)}")

    base_oof = pd.read_parquet(base_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if base_oof["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("base OOF rows do not match sealed development rows")
    base_scores = base_oof["score"].to_numpy(dtype=np.float64)

    result = crossfit_attribute_evidence_stack(
        items,
        dev_pairs,
        base_scores,
        fold_ids,
        min_support=min_support,
        smoothing=smoothing,
        evidence_clip=evidence_clip,
    )
    payload = {
        "version": "v5b-attribute-evidence",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev_pairs)),
        "development_items": int(len(items)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "gold_items_used": 0,
        "min_support": int(min_support),
        "smoothing": float(smoothing),
        "evidence_clip": float(evidence_clip),
        "base_oof_macro_ap": float(result["base_macro_average_precision"]),
        "attribute_oof_macro_ap": float(result["macro_average_precision"]),
        "delta_vs_base": float(result["delta_vs_base"]),
        "fold_reports": result["fold_reports"],
        "per_category_ap": result["per_category_ap"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "v5b-attribute-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": fold_ids,
            "evidence": result["evidence"],
            "score": result["scores"],
        }
    ).to_parquet(output_dir / "v5b-attribute-oof.parquet", index=False)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--smoothing", type=float, default=2.0)
    parser.add_argument("--evidence-clip", type=float, default=8.0)
    args = parser.parse_args()
    payload = run_attribute_oof(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        base_oof_path=args.base_oof,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
        min_support=args.min_support,
        smoothing=args.smoothing,
        evidence_clip=args.evidence_clip,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
