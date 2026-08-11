from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .features import normalize_items
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_category_specialists import fit_predict_category_specialists
from .v5_evaluation import macro_ap_report
from .v5_validation import manifest_sha256
from .v5_weighted_specialists import build_fold_weighted_features, fit_fold_attribute_importance


def run_weighted_fold(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    output_dir: Path,
    expected_split_sha: str,
    held_fold: int,
    min_support: int = 20,
    max_iter: int = 300,
    seed: int = 2026,
) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = manifest_sha256(manifest)
    if actual != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual}")

    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if held_fold not in set(fold_ids.tolist()):
        raise ValueError(f"held fold {held_fold} not present")
    dev_pairs = matches.iloc[dev_rows].reset_index(drop=True)
    train_mask = fold_ids != held_fold
    valid_mask = fold_ids == held_fold
    train_pairs = dev_pairs.loc[train_mask].reset_index(drop=True)
    valid_pairs = dev_pairs.loc[valid_mask].reset_index(drop=True)
    held_rows = dev_rows[valid_mask]

    wanted_ids = pd.unique(pd.concat([dev_pairs["id1"], dev_pairs["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=True)
    item_cache = normalize_items(items)
    category_by_id = items.set_index("id")["category"].astype(str)
    train_pairs["category"] = train_pairs["id1"].map(category_by_id)
    valid_pairs["category"] = valid_pairs["id1"].map(category_by_id)
    if train_pairs["category"].isna().any() or valid_pairs["category"].isna().any():
        raise RuntimeError("failed to attach categories")

    gold_rows = np.asarray(manifest["gold_rows"], dtype=np.int64)
    gold_pairs = matches.iloc[gold_rows]
    gold_ids = set(gold_pairs["id1"].tolist()) | set(gold_pairs["id2"].tolist())
    if gold_ids & set(item_cache):
        raise RuntimeError("gold items leaked into weighted specialist development set")

    importance_started = time.perf_counter()
    importance = fit_fold_attribute_importance(
        items,
        train_pairs[["id1", "id2", "target"]],
        min_support=min_support,
        item_cache=item_cache,
    )
    importance_seconds = time.perf_counter() - importance_started

    feature_started = time.perf_counter()
    train_features = build_fold_weighted_features(
        items,
        train_pairs[["id1", "id2"]],
        importance,
        item_cache=item_cache,
    )
    valid_features = build_fold_weighted_features(
        items,
        valid_pairs[["id1", "id2"]],
        importance,
        item_cache=item_cache,
    )
    feature_seconds = time.perf_counter() - feature_started

    score = fit_predict_category_specialists(
        train_features,
        train_pairs["target"].to_numpy(),
        valid_features,
        seed=seed + held_fold,
        max_iter=max_iter,
        min_samples_leaf=15,
        l2_regularization=2.0,
    )
    report = macro_ap_report(valid_pairs, score)
    output = pd.DataFrame(
        {
            "row_index": held_rows,
            "fold": np.full(len(held_rows), held_fold, dtype=np.int8),
            "score": score,
        }
    ).sort_values("row_index")
    output.to_parquet(output_dir / f"v5b-weighted-fold-{held_fold}-oof.parquet", index=False)

    payload = {
        "version": "v5b-weighted-category-specialist",
        "held_fold": int(held_fold),
        "split_sha256": expected_split_sha,
        "gold_metric_opened": False,
        "gold_rows_used": 0,
        "gold_items_used": 0,
        "train_rows": int(len(train_pairs)),
        "valid_rows": int(len(valid_pairs)),
        "learned_categories": int(len(importance)),
        "learned_keys": int(sum(len(v) for v in importance.values())),
        "min_support": int(min_support),
        "importance_seconds": float(importance_seconds),
        "feature_seconds": float(feature_seconds),
        "macro_average_precision": float(report["macro_average_precision"]),
        "per_category_ap": report["per_category_ap"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / f"v5b-weighted-fold-{held_fold}-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    parser.add_argument("--held-fold", type=int, required=True)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    payload = run_weighted_fold(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
        held_fold=args.held_fold,
        min_support=args.min_support,
        max_iter=args.max_iter,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
