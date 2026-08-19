from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .features import normalize_items
from .run_v5_fixed_blend import align_oof_frame
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_evaluation import macro_ap_report
from .v5_fixed_blend import percentile_rank
from .v5_typed_consistency import typed_quantity_consistency
from .v5_validation import manifest_sha256


ANCHOR_COLUMN = "candidate_current4_plus_teacher"
CURRENT5_COLUMNS = (
    "source_weak",
    "source_sparse",
    "source_explicit",
    "source_contrastive_cosine",
    "source_teacher2_raw",
)


def run_typed_consistency(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    anchor_oof_path: Path,
    output_dir: Path,
    expected_split_sha: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_split_sha = manifest_sha256(manifest)
    if actual_split_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_split_sha}")

    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, folds = development_rows_and_folds(manifest, total_rows=len(matches))
    folds = np.asarray(folds, dtype=np.int16)
    anchor = align_oof_frame(
        [anchor_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=(ANCHOR_COLUMN, *CURRENT5_COLUMNS),
        source_name="current5_anchor",
    )

    dev = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=True)
    item_cache = normalize_items(items)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach official categories to development rows")

    typed_scores = np.fromiter(
        (
            typed_quantity_consistency(item_cache[id1], item_cache[id2])
            for id1, id2 in dev[["id1", "id2"]].itertuples(index=False, name=None)
        ),
        dtype=np.float64,
        count=len(dev),
    )
    if not np.isfinite(typed_scores).all():
        raise RuntimeError("typed consistency produced non-finite scores")

    anchor_scores = anchor[ANCHOR_COLUMN].to_numpy(dtype=np.float64)
    anchor_report = macro_ap_report(dev, anchor_scores, strict_official=True)
    anchor_ap = float(anchor_report["macro_average_precision"])
    expected_anchor = 0.5952697490140912
    if abs(anchor_ap - expected_anchor) > 1e-12:
        raise RuntimeError(f"anchor mismatch: observed={anchor_ap}, expected={expected_anchor}")

    current_ranks = [
        percentile_rank(anchor[column].to_numpy(dtype=np.float64))
        for column in CURRENT5_COLUMNS
    ]
    candidate_scores = np.mean(
        np.vstack(current_ranks + [percentile_rank(typed_scores)]),
        axis=0,
    )
    typed_report = macro_ap_report(dev, typed_scores, strict_official=True)
    candidate_report = macro_ap_report(dev, candidate_scores, strict_official=True)
    candidate_ap = float(candidate_report["macro_average_precision"])

    fold_reports = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        fold_frame = dev.loc[mask].reset_index(drop=True)
        fold_anchor = float(macro_ap_report(fold_frame, anchor_scores[mask])["macro_average_precision"])
        fold_candidate = float(macro_ap_report(fold_frame, candidate_scores[mask])["macro_average_precision"])
        fold_reports.append(
            {
                "fold": int(fold),
                "rows": int(mask.sum()),
                "anchor_macro_average_precision": fold_anchor,
                "macro_average_precision": fold_candidate,
                "delta_vs_anchor": float(fold_candidate - fold_anchor),
            }
        )
    min_fold_delta = min(row["delta_vs_anchor"] for row in fold_reports)
    keep = bool(candidate_ap > anchor_ap and min_fold_delta >= -0.001)

    comparable = np.abs(typed_scores) > 0
    payload = {
        "version": "v5-target-free-typed-consistency",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "target_fitted_blender": False,
        "predeclared_candidate_count": 1,
        "anchor_macro_average_precision": anchor_ap,
        "typed_consistency_macro_average_precision": float(typed_report["macro_average_precision"]),
        "typed_consistency_nonzero_rows": int(comparable.sum()),
        "typed_consistency_nonzero_rate": float(comparable.mean()),
        "candidate_name": "current5_plus_typed_consistency",
        "candidate_macro_average_precision": candidate_ap,
        "delta_vs_anchor": float(candidate_ap - anchor_ap),
        "min_fold_delta_vs_anchor": float(min_fold_delta),
        "fold_reports": fold_reports,
        "per_category_ap": candidate_report["per_category_ap"],
        "keep_eligible": keep,
        "target_0_60_reached": bool(candidate_ap >= 0.60),
    }
    (output_dir / "v5-typed-consistency-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": folds,
            "typed_consistency_score": typed_scores,
            "candidate_score": candidate_scores,
        }
    ).to_parquet(output_dir / "v5-typed-consistency-oof.parquet", index=False)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    args = parser.parse_args()
    payload = run_typed_consistency(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        anchor_oof_path=args.anchor_oof,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
