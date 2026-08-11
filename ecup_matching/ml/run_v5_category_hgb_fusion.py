from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_evaluation import macro_ap_report
from .v5_fixed_blend import percentile_rank
from .v5_validation import manifest_sha256


EXPECTED_CATEGORY_SHRUNK_AP = 0.60095424180184
EXPECTED_HGB_AP = 0.6006290884983169


def _aligned_score(path: Path, column: str, rows: np.ndarray, folds: np.ndarray) -> np.ndarray:
    frame = pd.read_parquet(path)
    required = {"row_index", "fold", column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if not np.array_equal(frame["row_index"].to_numpy(dtype=np.int64), rows):
        raise ValueError(f"{path} row_index mismatch")
    if not np.array_equal(frame["fold"].to_numpy(dtype=np.int16), folds):
        raise ValueError(f"{path} fold mismatch")
    score = frame[column].to_numpy(dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError(f"{path} contains non-finite {column}")
    return score


def run_fusion(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    category_shrunk_oof_path: Path,
    hgb_oof_path: Path,
    output_dir: Path,
    expected_split_sha: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = manifest_sha256(manifest)
    if actual_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_sha}")

    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    rows, folds = development_rows_and_folds(manifest, total_rows=len(matches))
    rows = np.asarray(rows, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int16)
    category_score = _aligned_score(
        category_shrunk_oof_path, "category_shrunk_oof_score", rows, folds
    )
    hgb_score = _aligned_score(hgb_oof_path, "hgb_stack_oof_score", rows, folds)

    dev = matches.iloc[rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach official categories")

    category_ap = float(macro_ap_report(dev, category_score, strict_official=True)["macro_average_precision"])
    hgb_ap = float(macro_ap_report(dev, hgb_score, strict_official=True)["macro_average_precision"])
    if abs(category_ap - EXPECTED_CATEGORY_SHRUNK_AP) > 1e-12:
        raise RuntimeError(f"category anchor mismatch: {category_ap} != {EXPECTED_CATEGORY_SHRUNK_AP}")
    if abs(hgb_ap - EXPECTED_HGB_AP) > 1e-12:
        raise RuntimeError(f"hgb anchor mismatch: {hgb_ap} != {EXPECTED_HGB_AP}")

    fusion = 0.5 * percentile_rank(category_score) + 0.5 * percentile_rank(hgb_score)
    report = macro_ap_report(dev, fusion, strict_official=True)
    fusion_ap = float(report["macro_average_precision"])
    fold_reports = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        frame = dev.loc[mask].reset_index(drop=True)
        fold_reports.append(
            {
                "fold": int(fold),
                "category_shrunk_macro_ap": float(macro_ap_report(frame, category_score[mask])["macro_average_precision"]),
                "hgb_macro_ap": float(macro_ap_report(frame, hgb_score[mask])["macro_average_precision"]),
                "equal_rank_fusion_macro_ap": float(macro_ap_report(frame, fusion[mask])["macro_average_precision"]),
            }
        )

    payload = {
        "version": "v5-category-shrunk-hgb-equal-rank-fusion",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "fully_outer_cross_fitted_components": True,
        "post_result_weight_tuning": False,
        "fusion_formula": "0.5*percentile_rank(category_shrunk_oof)+0.5*percentile_rank(hgb_stack_oof)",
        "category_shrunk_macro_ap": category_ap,
        "hgb_stack_macro_ap": hgb_ap,
        "equal_rank_fusion_macro_ap": fusion_ap,
        "equal_rank_fusion_per_category_ap": report["per_category_ap"],
        "delta_vs_category_shrunk": float(fusion_ap - category_ap),
        "keep": bool(fusion_ap > category_ap),
        "target_0_60_reached": bool(fusion_ap >= 0.60),
        "fold_reports": fold_reports,
    }
    (output_dir / "v5-category-hgb-fusion-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": rows,
            "fold": folds,
            "category_shrunk_oof_score": category_score,
            "hgb_stack_oof_score": hgb_score,
            "equal_rank_fusion_score": fusion,
        }
    ).to_parquet(output_dir / "v5-category-hgb-fusion-oof.parquet", index=False)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--category-shrunk-oof", type=Path, required=True)
    parser.add_argument("--hgb-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    args = parser.parse_args()
    run_fusion(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        category_shrunk_oof_path=args.category_shrunk_oof,
        hgb_oof_path=args.hgb_oof,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
