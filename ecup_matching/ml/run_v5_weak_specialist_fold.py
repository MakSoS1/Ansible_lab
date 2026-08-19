from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .features_v2 import build_features_v2_chunked
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .train_v1 import attach_pair_category
from .train_v2_structured import candidate_sample_weights, prefilter_weak_candidates_parquet
from .v5_category_specialists import fit_predict_category_specialists
from .v5_evaluation import macro_ap_report
from .v5_validation import manifest_sha256
from .v5_weak_specialists import forbidden_weak_item_ids
from .weak_labels import prepare_weak_pairs, remove_human_conflicts, sample_weak_training


def run_weak_specialist_fold(
    *,
    human_items_path: Path,
    matches_path: Path,
    weak_matches_path: Path,
    full_items_path: Path,
    manifest_path: Path,
    output_dir: Path,
    expected_split_sha: str,
    held_fold: int,
    weak_presample_rows: int = 250_000,
    weak_final_rows: int = 150_000,
    chunk_size: int = 25_000,
    max_iter: int = 300,
    seed: int = 2026,
) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = manifest_sha256(manifest)
    if actual_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_sha}")

    human_items = pd.read_parquet(
        human_items_path, columns=["id", "name", "attributes", "category"]
    )
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, fold_ids = development_rows_and_folds(manifest, total_rows=len(matches))
    if held_fold not in set(fold_ids.tolist()):
        raise ValueError(f"held fold {held_fold} not present")

    dev_pairs = matches.iloc[dev_rows].reset_index(drop=True)
    train_mask = fold_ids != held_fold
    valid_mask = fold_ids == held_fold
    human_train = dev_pairs.loc[train_mask, ["id1", "id2", "target"]].reset_index(drop=True)
    valid_pairs = dev_pairs.loc[valid_mask, ["id1", "id2", "target"]].reset_index(drop=True)
    held_rows = dev_rows[valid_mask]

    human_train = attach_pair_category(human_train, human_items)
    valid_pairs = attach_pair_category(valid_pairs, human_items)

    forbidden = forbidden_weak_item_ids(matches, manifest, held_fold=held_fold)
    weak_started = time.perf_counter()
    weak, weak_input_rows = prefilter_weak_candidates_parquet(
        weak_matches_path,
        validation_item_ids=forbidden,
        max_presample_rows=weak_presample_rows,
        seed=seed + held_fold,
    )
    weak, prepare_report = prepare_weak_pairs(weak[["id1", "id2", "target"]])
    weak, conflict_report = remove_human_conflicts(
        weak,
        human_train[["id1", "id2", "target"]],
    )
    weak_ids = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    weak_items = select_items_by_ids(full_items_path, weak_ids, include_attributes=True)
    weak = attach_pair_category(weak, weak_items)
    weak = sample_weak_training(
        weak,
        max_rows=weak_final_rows,
        seed=seed + held_fold,
    )
    final_weak_ids = set(weak["id1"].tolist()) | set(weak["id2"].tolist())
    if final_weak_ids & forbidden:
        raise RuntimeError("weak curriculum contains held-fold or sealed-gold item")
    weak_items = weak_items[weak_items["id"].isin(final_weak_ids)].reset_index(drop=True)
    weak_selection_seconds = time.perf_counter() - weak_started

    feature_started = time.perf_counter()
    x_human = build_features_v2_chunked(
        human_items,
        human_train,
        attribute_importance=None,
        chunk_size=chunk_size,
    )
    x_valid = build_features_v2_chunked(
        human_items,
        valid_pairs,
        attribute_importance=None,
        chunk_size=chunk_size,
    )
    x_weak = build_features_v2_chunked(
        weak_items,
        weak,
        attribute_importance=None,
        chunk_size=chunk_size,
    )
    feature_seconds = time.perf_counter() - feature_started

    x_combined = pd.concat([x_human, x_weak], ignore_index=True)
    y_combined = np.concatenate(
        [
            human_train["target"].to_numpy(dtype=np.int8),
            weak["hard_target"].to_numpy(dtype=np.int8),
        ]
    )
    category_combined = pd.concat(
        [
            human_train["category"].reset_index(drop=True),
            weak["category"].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    source_combined = pd.Series(
        ["human"] * len(human_train) + ["weak"] * len(weak)
    )
    weak_weight_combined = np.concatenate(
        [
            np.ones(len(human_train), dtype=np.float64),
            weak["weak_weight"].to_numpy(dtype=np.float64),
        ]
    )
    weights = candidate_sample_weights(
        category_combined,
        source_combined,
        y_combined,
        weak_weight_combined,
        x_combined["hard_negative_score"].to_numpy(dtype=np.float64),
        hard_negative_boost=0.0,
    )

    fit_started = time.perf_counter()
    score = fit_predict_category_specialists(
        x_combined,
        y_combined,
        x_valid,
        sample_weight=weights,
        seed=seed + held_fold,
        max_iter=max_iter,
        min_samples_leaf=15,
        l2_regularization=2.0,
    )
    fit_seconds = time.perf_counter() - fit_started
    report = macro_ap_report(valid_pairs, score)

    pd.DataFrame(
        {
            "row_index": held_rows,
            "fold": np.full(len(held_rows), held_fold, dtype=np.int8),
            "score": score,
        }
    ).sort_values("row_index").to_parquet(
        output_dir / f"v5e-weak-fold-{held_fold}-oof.parquet", index=False
    )

    payload = {
        "version": "v5e-weak-category-specialist-sprint",
        "held_fold": int(held_fold),
        "split_sha256": expected_split_sha,
        "gold_metric_opened": False,
        "gold_rows_used": 0,
        "gold_items_used": 0,
        "held_items_forbidden_from_weak": True,
        "human_train_rows": int(len(human_train)),
        "valid_rows": int(len(valid_pairs)),
        "weak_input_rows": int(weak_input_rows),
        "weak_presample_cap": int(weak_presample_rows),
        "weak_final_rows": int(len(weak)),
        "weak_unique_items": int(len(final_weak_ids)),
        "weak_prepare": prepare_report,
        "weak_conflicts": conflict_report,
        "weak_selection_seconds": float(weak_selection_seconds),
        "feature_seconds": float(feature_seconds),
        "fit_seconds": float(fit_seconds),
        "macro_average_precision": float(report["macro_average_precision"]),
        "per_category_ap": report["per_category_ap"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output_dir / f"v5e-weak-fold-{held_fold}-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--weak-matches", type=Path, required=True)
    parser.add_argument("--full-items", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    parser.add_argument("--held-fold", type=int, required=True)
    parser.add_argument("--weak-presample-rows", type=int, default=250_000)
    parser.add_argument("--weak-final-rows", type=int, default=150_000)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    payload = run_weak_specialist_fold(
        human_items_path=args.human_items,
        matches_path=args.matches,
        weak_matches_path=args.weak_matches,
        full_items_path=args.full_items,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
        held_fold=args.held_fold,
        weak_presample_rows=args.weak_presample_rows,
        weak_final_rows=args.weak_final_rows,
        chunk_size=args.chunk_size,
        max_iter=args.max_iter,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
