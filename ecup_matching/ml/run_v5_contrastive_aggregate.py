from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_embeddings import EMBEDDING_PAIR_FEATURE_NAMES
from .v5_evaluation import macro_ap_report
from .v5_semantic_stack import crossfit_semantic_stack
from .v5_validation import manifest_sha256


def aggregate_contrastive_oof(
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

    pieces = [pd.read_parquet(path) for path in fold_files]
    if len(pieces) < 2:
        raise ValueError("at least two fold OOF files are required")
    semantic = pd.concat(pieces, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    if semantic["row_index"].duplicated().any():
        raise ValueError("duplicate OOF row across contrastive fold files")
    if semantic["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("contrastive OOF rows do not exactly cover sealed development rows")
    if semantic["fold"].astype(np.int16).to_numpy().tolist() != expected_folds.tolist():
        raise ValueError("contrastive OOF fold IDs do not match sealed manifest")

    missing_features = set(EMBEDDING_PAIR_FEATURE_NAMES) - set(semantic.columns)
    if missing_features:
        raise ValueError(f"contrastive OOF missing semantic features: {sorted(missing_features)}")

    dev_pairs = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev_pairs["id1"], dev_pairs["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev_pairs["category"] = dev_pairs["id1"].map(category_by_id)
    if dev_pairs["category"].isna().any():
        raise RuntimeError("failed to attach categories")

    category_oof = pd.read_parquet(category_oof_path, columns=["row_index", "score"]).sort_values("row_index")
    if category_oof["row_index"].astype(np.int64).tolist() != dev_rows.tolist():
        raise ValueError("category-specialist OOF rows do not match development rows")
    base_scores = category_oof["score"].to_numpy(dtype=np.float64)
    semantic_features = semantic.loc[:, EMBEDDING_PAIR_FEATURE_NAMES].astype(np.float32)
    fold_ids = semantic["fold"].to_numpy(dtype=np.int16)

    cosine_report = macro_ap_report(dev_pairs, semantic_features["embedding_cosine"].to_numpy())
    stack = crossfit_semantic_stack(
        dev_pairs,
        base_scores,
        semantic_features,
        fold_ids,
        seed=2026,
        max_iter=220,
    )
    payload = {
        "version": "v5d-contrastive-human-sprint",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev_pairs)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "category_base_oof_macro_ap": float(stack["base_macro_average_precision"]),
        "contrastive_cosine_oof_macro_ap": float(cosine_report["macro_average_precision"]),
        "stack_oof_macro_ap": float(stack["macro_average_precision"]),
        "delta_vs_category_base": float(stack["delta_vs_base"]),
        "fold_reports": stack["fold_reports"],
        "per_category_ap": stack["per_category_ap"],
    }
    (output_dir / "v5d-contrastive-sprint-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": fold_ids,
            **{name: semantic_features[name].to_numpy() for name in EMBEDDING_PAIR_FEATURE_NAMES},
            "score": stack["scores"],
        }
    ).to_parquet(output_dir / "v5d-contrastive-sprint-oof.parquet", index=False)
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
    fold_files = sorted(args.fold_dir.rglob("v5d-fold-*-oof.parquet"))
    payload = aggregate_contrastive_oof(
        fold_files=fold_files,
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
