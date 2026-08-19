from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_evaluation import macro_ap_report
from .v5_fixed_blend import percentile_rank
from .v5_validation import manifest_sha256


EXPECTED_GLOBAL_META_AP = 0.5992720660193247
EXPECTED_CATEGORY_LOGISTIC_AP = 0.5988060044248327


def _peak_ram_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _aligned_oof(path: Path, *, score_column: str, rows: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"row_index", "fold", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(frame) != len(rows):
        raise ValueError(f"{path} row count mismatch: {len(frame)} != {len(rows)}")
    if not np.array_equal(frame["row_index"].to_numpy(dtype=np.int64), rows):
        raise ValueError(f"{path} row_index mismatch")
    if not np.array_equal(frame["fold"].to_numpy(dtype=np.int16), folds):
        raise ValueError(f"{path} fold mismatch")
    score = frame[score_column].to_numpy(dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError(f"{path} {score_column} must be finite")
    return frame


def run_frozen_fusion(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    global_meta_oof_path: Path,
    category_logistic_oof_path: Path,
    output_dir: Path,
    expected_split_sha: str,
) -> dict:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = manifest_sha256(manifest)
    if actual_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_sha}")

    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, folds = development_rows_and_folds(manifest, total_rows=len(matches))
    dev_rows = np.asarray(dev_rows, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int16)

    global_meta = _aligned_oof(
        global_meta_oof_path,
        score_column="meta_oof_score",
        rows=dev_rows,
        folds=folds,
    )
    category_logistic = _aligned_oof(
        category_logistic_oof_path,
        score_column="category_logistic_oof_score",
        rows=dev_rows,
        folds=folds,
    )

    dev = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach official categories")

    global_score = global_meta["meta_oof_score"].to_numpy(dtype=np.float64)
    logistic_score = category_logistic["category_logistic_oof_score"].to_numpy(dtype=np.float64)
    global_report = macro_ap_report(dev, global_score, strict_official=True)
    logistic_report = macro_ap_report(dev, logistic_score, strict_official=True)
    global_ap = float(global_report["macro_average_precision"])
    logistic_ap = float(logistic_report["macro_average_precision"])
    if abs(global_ap - EXPECTED_GLOBAL_META_AP) > 1e-12:
        raise RuntimeError(f"global meta anchor mismatch: {global_ap} != {EXPECTED_GLOBAL_META_AP}")
    if abs(logistic_ap - EXPECTED_CATEGORY_LOGISTIC_AP) > 1e-12:
        raise RuntimeError(
            f"category logistic anchor mismatch: {logistic_ap} != {EXPECTED_CATEGORY_LOGISTIC_AP}"
        )

    fusion = 0.5 * percentile_rank(global_score) + 0.5 * percentile_rank(logistic_score)
    fusion_report = macro_ap_report(dev, fusion, strict_official=True)
    fusion_ap = float(fusion_report["macro_average_precision"])

    fold_reports = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        frame = dev.loc[mask].reset_index(drop=True)
        fold_reports.append(
            {
                "fold": int(fold),
                "rows": int(mask.sum()),
                "global_meta_macro_ap": float(
                    macro_ap_report(frame, global_score[mask])["macro_average_precision"]
                ),
                "category_logistic_macro_ap": float(
                    macro_ap_report(frame, logistic_score[mask])["macro_average_precision"]
                ),
                "frozen_equal_rank_fusion_macro_ap": float(
                    macro_ap_report(frame, fusion[mask])["macro_average_precision"]
                ),
            }
        )

    payload = {
        "version": "v5-frozen-global-meta-category-logistic-equal-rank",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "fully_outer_cross_fitted_components": True,
        "post_result_weight_tuning": False,
        "fusion_formula": "0.5*percentile_rank(global_meta_oof)+0.5*percentile_rank(category_logistic_oof)",
        "global_meta_macro_ap": global_ap,
        "category_logistic_macro_ap": logistic_ap,
        "frozen_equal_rank_fusion_macro_ap": fusion_ap,
        "frozen_equal_rank_fusion_per_category_ap": fusion_report["per_category_ap"],
        "delta_vs_global_meta": float(fusion_ap - global_ap),
        "target_0_60_reached": bool(fusion_ap >= 0.60),
        "fold_reports": fold_reports,
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_ram_gib": _peak_ram_gib(),
    }
    (output_dir / "v5-meta-logistic-fusion-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": folds,
            "global_meta_oof_score": global_score,
            "category_logistic_oof_score": logistic_score,
            "frozen_equal_rank_fusion_score": fusion,
        }
    ).to_parquet(output_dir / "v5-meta-logistic-fusion-oof.parquet", index=False)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--global-meta-oof", type=Path, required=True)
    parser.add_argument("--category-logistic-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    args = parser.parse_args()
    run_frozen_fusion(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        global_meta_oof_path=args.global_meta_oof,
        category_logistic_oof_path=args.category_logistic_oof,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
