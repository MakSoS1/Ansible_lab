from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .features_v2 import build_features_v2_chunked
from .v5_category_specialists import fit_predict_category_specialists
from .v5_evaluation import macro_ap_report
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_validation import manifest_sha256


def run_category_oof(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    base_oof_path: Path,
    output_dir: Path,
    expected_split_sha: str,
    chunk_size: int = 25_000,
    max_iter: int = 300,
) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_sha256(manifest) != expected_split_sha:
        raise ValueError("sealed split SHA mismatch")

    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    dev_pairs = matches.iloc[dev_rows].reset_index(drop=True)
    dev_item_ids = pd.unique(pd.concat([dev_pairs["id1"], dev_pairs["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, dev_item_ids, include_attributes=True)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev_pairs["category"] = dev_pairs["id1"].map(category_by_id)
    if dev_pairs["category"].isna().any():
        raise RuntimeError("failed to attach development categories")

    feature_started = time.perf_counter()
    features = build_features_v2_chunked(items, dev_pairs, attribute_importance=None, chunk_size=chunk_size)
    feature_seconds = time.perf_counter() - feature_started

    scores = np.full(len(dev_pairs), np.nan, dtype=np.float64)
    fold_reports: list[dict] = []
    for fold in sorted(np.unique(fold_ids).tolist()):
        valid_mask = fold_ids == fold
        train_mask = ~valid_mask
        fold_score = fit_predict_category_specialists(
            features.loc[train_mask].reset_index(drop=True),
            dev_pairs.loc[train_mask, "target"].to_numpy(),
            features.loc[valid_mask].reset_index(drop=True),
            seed=2026 + int(fold),
            max_iter=max_iter,
            min_samples_leaf=15,
            l2_regularization=2.0,
        )
        scores[valid_mask] = fold_score
        report = macro_ap_report(dev_pairs.loc[valid_mask].reset_index(drop=True), fold_score)
        fold_reports.append(
            {
                "fold": int(fold),
                "valid_rows": int(valid_mask.sum()),
                "macro_average_precision": float(report["macro_average_precision"]),
            }
        )

    if not np.isfinite(scores).all():
        raise RuntimeError("missing category-specialist OOF scores")
    candidate = macro_ap_report(dev_pairs, scores)
    base_oof = pd.read_parquet(base_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if base_oof["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("base OOF row indices do not match sealed development rows")
    base_scores = base_oof["score"].to_numpy(dtype=np.float64)
    base = macro_ap_report(dev_pairs, base_scores)

    payload = {
        "version": "v5b-category-specialists",
        "development_rows": int(len(dev_pairs)),
        "development_items": int(len(items)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "split_sha256": expected_split_sha,
        "feature_seconds": float(feature_seconds),
        "base_oof_macro_ap": float(base["macro_average_precision"]),
        "category_specialist_oof_macro_ap": float(candidate["macro_average_precision"]),
        "delta_vs_base": float(candidate["macro_average_precision"] - base["macro_average_precision"]),
        "fold_reports": fold_reports,
        "per_category_ap": candidate["per_category_ap"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "v5b-category-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame({"row_index": dev_rows, "fold": fold_ids, "score": scores}).to_parquet(
        output_dir / "v5b-category-oof.parquet", index=False
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_category_oof(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        base_oof_path=args.base_oof,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
        chunk_size=args.chunk_size,
        max_iter=args.max_iter,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
