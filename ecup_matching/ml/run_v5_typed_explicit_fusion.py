from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_fixed_blend import align_oof_frame
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_evaluation import macro_ap_report
from .v5_fixed_blend import typed_explicit_rank_candidates
from .v5_validation import manifest_sha256


ANCHOR_COLUMN = "candidate_current4_plus_teacher"
CURRENT5_COLUMNS = {
    "weak": "source_weak",
    "sparse": "source_sparse",
    "explicit": "source_explicit",
    "contrastive": "source_contrastive_cosine",
    "teacher": "source_teacher2_raw",
}


def run_typed_explicit_fusion(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    anchor_oof_path: Path,
    typed_fold_dir: Path,
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
        required_columns=(ANCHOR_COLUMN, *CURRENT5_COLUMNS.values()),
        source_name="current5_anchor",
    )
    typed = align_oof_frame(
        sorted(Path(typed_fold_dir).rglob("v5-explicit-fold-*-oof.parquet")),
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("score",),
        source_name="typed_explicit",
    )

    dev = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach official categories to development rows")

    anchor_scores = anchor[ANCHOR_COLUMN].to_numpy(dtype=np.float64)
    anchor_ap = float(macro_ap_report(dev, anchor_scores, strict_official=True)["macro_average_precision"])
    expected_anchor = 0.5952697490140912
    if abs(anchor_ap - expected_anchor) > 1e-12:
        raise RuntimeError(f"current5 anchor mismatch: observed={anchor_ap}, expected={expected_anchor}")

    current5 = {
        name: anchor[column].to_numpy(dtype=np.float64)
        for name, column in CURRENT5_COLUMNS.items()
    }
    typed_scores = typed["score"].to_numpy(dtype=np.float64)
    typed_direct = macro_ap_report(dev, typed_scores, strict_official=True)
    candidates = typed_explicit_rank_candidates(current5, typed_scores)

    reports: dict[str, dict] = {}
    eligible: list[str] = []
    for name, scores in candidates.items():
        report = macro_ap_report(dev, scores, strict_official=True)
        fold_reports = []
        for fold in sorted(np.unique(folds).tolist()):
            mask = folds == fold
            fold_frame = dev.loc[mask].reset_index(drop=True)
            fold_anchor = float(macro_ap_report(fold_frame, anchor_scores[mask])["macro_average_precision"])
            fold_candidate = float(macro_ap_report(fold_frame, scores[mask])["macro_average_precision"])
            fold_reports.append({
                "fold": int(fold),
                "rows": int(mask.sum()),
                "anchor_macro_average_precision": fold_anchor,
                "macro_average_precision": fold_candidate,
                "delta_vs_anchor": float(fold_candidate - fold_anchor),
            })
        candidate_ap = float(report["macro_average_precision"])
        min_fold_delta = min(row["delta_vs_anchor"] for row in fold_reports)
        keep = bool(candidate_ap > anchor_ap and min_fold_delta >= -0.001)
        if keep:
            eligible.append(name)
        reports[name] = {
            "macro_average_precision": candidate_ap,
            "delta_vs_anchor": float(candidate_ap - anchor_ap),
            "min_fold_delta_vs_anchor": float(min_fold_delta),
            "keep_eligible": keep,
            "fold_reports": fold_reports,
            "per_category_ap": report["per_category_ap"],
        }

    best_name = max(reports, key=lambda key: reports[key]["macro_average_precision"])
    best_ap = float(reports[best_name]["macro_average_precision"])
    payload = {
        "version": "v5-typed-explicit-fusion",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "target_fitted_blender": False,
        "predeclared_candidate_count": 2,
        "anchor_macro_average_precision": anchor_ap,
        "old_explicit_macro_average_precision": 0.5683065131240066,
        "typed_explicit_macro_average_precision": float(typed_direct["macro_average_precision"]),
        "typed_explicit_per_category_ap": typed_direct["per_category_ap"],
        "candidates": reports,
        "keep_eligible_candidates": eligible,
        "best_observed_name": best_name,
        "best_observed_macro_average_precision": best_ap,
        "best_observed_delta_vs_anchor": float(best_ap - anchor_ap),
        "target_0_60_reached": bool(best_ap >= 0.60),
    }
    (output_dir / "v5-typed-explicit-fusion-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame({
        "row_index": dev_rows,
        "fold": folds,
        "anchor_score": anchor_scores,
        "typed_explicit_score": typed_scores,
        **{f"candidate_{name}": score for name, score in candidates.items()},
    }).to_parquet(output_dir / "v5-typed-explicit-fusion-oof.parquet", index=False)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-oof", type=Path, required=True)
    parser.add_argument("--typed-fold-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    args = parser.parse_args()
    payload = run_typed_explicit_fusion(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        anchor_oof_path=args.anchor_oof,
        typed_fold_dir=args.typed_fold_dir,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
