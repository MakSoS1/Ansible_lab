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
from .v5_hgb_stack import DEFAULT_HGB_PARAMS, crossfit_fixed_hgb_stack
from .v5_meta_blend import SIX_SIGNAL_NAMES
from .v5_validation import manifest_sha256


EXPECTED_EQUAL_SIX_AP = 0.5975445721449741
STRICT_GLOBAL_META_AP = 0.5992720660193247
CURRENT_FROZEN_FUSION_AP = 0.5995921709945611


def _peak_ram_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def run_v5_hgb_stack(
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
    actual_sha = manifest_sha256(manifest)
    if actual_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_sha}")

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
        raise RuntimeError("failed to attach official categories")

    current5 = {
        name: anchor[column].to_numpy(dtype=np.float64)
        for name, column in CURRENT5_COLUMNS.items()
    }
    typed_score = typed["typed_explicit_score"].to_numpy(dtype=np.float64)
    six_scores = {**current5, "typed_explicit": typed_score}
    if tuple(six_scores) != SIX_SIGNAL_NAMES:
        raise RuntimeError(f"six-signal order mismatch: {tuple(six_scores)}")

    equal_six = typed_explicit_rank_candidates(current5, typed_score)["current5_plus_typed_explicit"]
    equal_report = macro_ap_report(dev, equal_six, strict_official=True)
    equal_ap = float(equal_report["macro_average_precision"])
    if abs(equal_ap - EXPECTED_EQUAL_SIX_AP) > 1e-12:
        raise RuntimeError(f"equal-six anchor mismatch: {equal_ap} != {EXPECTED_EQUAL_SIX_AP}")
    print(
        f"[hgb-stack] phase=anchor_verified score={equal_ap:.12f} rows={len(dev)} "
        f"elapsed={time.monotonic()-started:.1f}s peak_ram_gib={_peak_ram_gib():.3f}",
        flush=True,
    )

    crossfit_started = time.monotonic()
    def progress(done: int, total: int, fold: int) -> None:
        elapsed = time.monotonic() - crossfit_started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("inf")
        print(
            f"[hgb-stack] phase=outer_crossfit done={done}/{total} pct={100.0*done/total:.1f} "
            f"fold={fold} elapsed={elapsed:.1f}s rolling_folds_per_s={rate:.5f} "
            f"eta={eta:.1f}s peak_ram_gib={_peak_ram_gib():.3f}",
            flush=True,
        )

    result = crossfit_fixed_hgb_stack(
        six_scores,
        dev["target"].to_numpy(dtype=np.int8),
        dev["category"].to_numpy(dtype=str),
        folds,
        progress=progress,
        **DEFAULT_HGB_PARAMS,
    )
    oof = np.asarray(result["oof_score"], dtype=np.float64)
    report = macro_ap_report(dev, oof, strict_official=True)
    macro_ap = float(report["macro_average_precision"])
    fold_reports = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        frame = dev.loc[mask].reset_index(drop=True)
        fold_reports.append(
            {
                "fold": int(fold),
                "rows": int(mask.sum()),
                "hgb_stack_macro_ap": float(macro_ap_report(frame, oof[mask])["macro_average_precision"]),
                "equal_six_macro_ap": float(macro_ap_report(frame, equal_six[mask])["macro_average_precision"]),
            }
        )

    payload = {
        "version": "v5-fixed-nonlinear-hgb-meta-stack",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "fully_outer_cross_fitted": True,
        "target_fitted_blender": True,
        "hyperparameter_search": False,
        "category_balanced_training_loss": True,
        "signal_names": list(SIX_SIGNAL_NAMES),
        "hgb_params": result["params"],
        "equal_six_anchor_macro_ap": equal_ap,
        "strict_global_meta_macro_ap_reference": STRICT_GLOBAL_META_AP,
        "current_frozen_fusion_macro_ap_reference": CURRENT_FROZEN_FUSION_AP,
        "strict_hgb_stack_macro_ap": macro_ap,
        "strict_hgb_stack_per_category_ap": report["per_category_ap"],
        "delta_vs_strict_global_meta": float(macro_ap - STRICT_GLOBAL_META_AP),
        "delta_vs_current_frozen_fusion": float(macro_ap - CURRENT_FROZEN_FUSION_AP),
        "target_0_60_reached": bool(macro_ap >= 0.60),
        "fold_reports": fold_reports,
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_ram_gib": _peak_ram_gib(),
    }
    (output_dir / "v5-hgb-stack-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {"row_index": dev_rows, "fold": folds, "hgb_stack_oof_score": oof}
    ).to_parquet(output_dir / "v5-hgb-stack-oof.parquet", index=False)
    (output_dir / "v5-hgb-stack-timing.json").write_text(
        json.dumps(
            {"elapsed_seconds": payload["elapsed_seconds"], "peak_ram_gib": payload["peak_ram_gib"], "rows": len(dev), "folds": len(fold_reports)},
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
    run_v5_hgb_stack(
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
