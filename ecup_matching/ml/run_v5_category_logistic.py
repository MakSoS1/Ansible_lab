from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_fixed_blend import align_oof_frame
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v5_typed_explicit_fusion import CURRENT5_COLUMNS
from .v5_evaluation import macro_ap_report
from .v5_fixed_blend import typed_explicit_rank_candidates
from .v5_meta_blend import SIX_SIGNAL_NAMES, crossfit_nested_category_logistic
from .v5_validation import manifest_sha256


EXPECTED_EQUAL_SIX_AP = 0.5975445721449741
STRICT_GLOBAL_META_AP = 0.5992720660193247
DEFAULT_C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)


def _peak_ram_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def run_v5_category_logistic(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    anchor_oof_path: Path,
    typed_fusion_oof_path: Path,
    output_dir: Path,
    expected_split_sha: str,
) -> dict:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_split_sha = manifest_sha256(manifest)
    if actual_split_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_split_sha}")

    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, folds = development_rows_and_folds(manifest, total_rows=len(matches))
    dev_rows = np.asarray(dev_rows, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int16)

    anchor = align_oof_frame(
        [anchor_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=tuple(CURRENT5_COLUMNS.values()),
        source_name="current5_anchor",
    )
    typed = align_oof_frame(
        [typed_fusion_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("typed_explicit_score",),
        source_name="typed_explicit_fusion",
    )

    dev = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach official categories to development rows")

    current5 = {
        name: anchor[column].to_numpy(dtype=np.float64)
        for name, column in CURRENT5_COLUMNS.items()
    }
    typed_scores = typed["typed_explicit_score"].to_numpy(dtype=np.float64)
    six_scores = {**current5, "typed_explicit": typed_scores}
    if tuple(six_scores) != SIX_SIGNAL_NAMES:
        raise RuntimeError(f"six-signal order mismatch: {tuple(six_scores)} != {SIX_SIGNAL_NAMES}")

    equal_six = typed_explicit_rank_candidates(current5, typed_scores)["current5_plus_typed_explicit"]
    equal_six_report = macro_ap_report(dev, equal_six, strict_official=True)
    equal_six_ap = float(equal_six_report["macro_average_precision"])
    if abs(equal_six_ap - EXPECTED_EQUAL_SIX_AP) > 1e-12:
        raise RuntimeError(
            f"equal-six anchor mismatch: observed={equal_six_ap}, expected={EXPECTED_EQUAL_SIX_AP}"
        )
    print(
        f"[category-logistic] phase=anchor_verified score={equal_six_ap:.12f} rows={len(dev)} "
        f"elapsed={time.monotonic() - started:.1f}s peak_ram_gib={_peak_ram_gib():.3f}",
        flush=True,
    )

    crossfit_started = time.monotonic()

    def progress(done: int, total: int, fold: int, chosen_c: float, inner_ap: float) -> None:
        elapsed = time.monotonic() - crossfit_started
        throughput = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / throughput if throughput > 0 else float("inf")
        print(
            f"[category-logistic] phase=nested_outer done={done}/{total} pct={100.0*done/total:.1f} "
            f"fold={fold} selected_c={chosen_c:g} inner_macro_ap={inner_ap:.9f} "
            f"elapsed={elapsed:.1f}s rolling_folds_per_s={throughput:.5f} eta={eta:.1f}s "
            f"peak_ram_gib={_peak_ram_gib():.3f}",
            flush=True,
        )

    nested = crossfit_nested_category_logistic(
        six_scores,
        dev["target"].to_numpy(dtype=np.int8),
        dev["category"].to_numpy(dtype=str),
        folds,
        c_grid=DEFAULT_C_GRID,
        max_iter=250,
        progress=progress,
    )
    logistic_oof = np.asarray(nested["oof_score"], dtype=np.float64)
    report = macro_ap_report(dev, logistic_oof, strict_official=True)
    macro_ap = float(report["macro_average_precision"])

    fold_reports = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        frame = dev.loc[mask].reset_index(drop=True)
        equal_fold = float(macro_ap_report(frame, equal_six[mask])["macro_average_precision"])
        logistic_fold = float(macro_ap_report(frame, logistic_oof[mask])["macro_average_precision"])
        fold_reports.append(
            {
                "fold": int(fold),
                "rows": int(mask.sum()),
                "selected_c": float(nested["selected_c"][int(fold)]),
                "inner_macro_ap_by_c": {
                    str(c): float(score)
                    for c, score in nested["inner_macro_ap_by_c"][int(fold)].items()
                },
                "equal_six_macro_average_precision": equal_fold,
                "category_logistic_macro_average_precision": logistic_fold,
                "delta_vs_equal_six": float(logistic_fold - equal_fold),
            }
        )

    elapsed_total = time.monotonic() - started
    payload = {
        "version": "v5-fully-nested-category-logistic",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "target_fitted_blender": True,
        "fully_outer_cross_fitted": True,
        "fully_nested_hyperparameter_selection": True,
        "category_balanced_training_loss": True,
        "feature_family": "six global percentile ranks + category-by-rank interactions",
        "signal_names": list(SIX_SIGNAL_NAMES),
        "c_grid": [float(v) for v in DEFAULT_C_GRID],
        "equal_six_anchor_macro_ap": equal_six_ap,
        "strict_global_meta_macro_ap_reference": STRICT_GLOBAL_META_AP,
        "strict_nested_category_logistic_macro_ap": macro_ap,
        "strict_nested_category_logistic_per_category_ap": report["per_category_ap"],
        "delta_vs_equal_six_anchor": float(macro_ap - equal_six_ap),
        "delta_vs_strict_global_meta": float(macro_ap - STRICT_GLOBAL_META_AP),
        "selected_c_by_outer_fold": {
            str(fold): float(value) for fold, value in nested["selected_c"].items()
        },
        "fold_reports": fold_reports,
        "target_0_60_reached": bool(macro_ap >= 0.60),
        "elapsed_seconds": float(elapsed_total),
        "peak_ram_gib": _peak_ram_gib(),
    }

    (output_dir / "v5-category-logistic-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": folds,
            "equal_six_score": equal_six,
            "category_logistic_oof_score": logistic_oof,
        }
    ).to_parquet(output_dir / "v5-category-logistic-oof.parquet", index=False)
    (output_dir / "v5-category-logistic-timing.json").write_text(
        json.dumps(
            {
                "version": payload["version"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "peak_ram_gib": payload["peak_ram_gib"],
                "folds": len(fold_reports),
                "rows": len(dev),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-oof", type=Path, required=True)
    parser.add_argument("--typed-fusion-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    args = parser.parse_args()
    run_v5_category_logistic(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        anchor_oof_path=args.anchor_oof,
        typed_fusion_oof_path=args.typed_fusion_oof,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
