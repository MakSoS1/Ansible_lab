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
from .v5_meta_blend import SIX_SIGNAL_NAMES, crossfit_global_simplex_blend, fit_simplex_weights
from .v5_validation import manifest_sha256


EXPECTED_EQUAL_SIX_AP = 0.5975445721449741
DEFAULT_STEP_SCHEDULE = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0, 1.0 / 96.0)


def _peak_ram_gib() -> float:
    # Linux ru_maxrss is KiB. GitHub runners for this workflow are Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def run_v5_meta_blend(
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

    print(f"[meta] phase=load_manifest elapsed={time.monotonic() - started:.1f}s peak_ram_gib={_peak_ram_gib():.3f}", flush=True)
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
        f"[meta] phase=anchor_verified score={equal_six_ap:.12f} rows={len(dev)} "
        f"elapsed={time.monotonic() - started:.1f}s peak_ram_gib={_peak_ram_gib():.3f}",
        flush=True,
    )

    fold_started = time.monotonic()

    def progress(done: int, total: int, train_rows: int, valid_rows: int) -> None:
        elapsed = time.monotonic() - fold_started
        throughput = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / throughput if throughput > 0 else float("inf")
        pct = 100.0 * done / total
        print(
            f"[meta] phase=outer_crossfit done={done}/{total} pct={pct:.1f} "
            f"train_rows={train_rows} valid_rows={valid_rows} elapsed={elapsed:.1f}s "
            f"rolling_folds_per_s={throughput:.4f} eta={eta:.1f}s peak_ram_gib={_peak_ram_gib():.3f}",
            flush=True,
        )

    crossfit = crossfit_global_simplex_blend(
        six_scores,
        dev["target"].to_numpy(dtype=np.int8),
        dev["category"].to_numpy(dtype=str),
        folds,
        step_schedule=DEFAULT_STEP_SCHEDULE,
        max_passes=6,
        progress=progress,
    )
    meta_oof = np.asarray(crossfit["oof_score"], dtype=np.float64)
    meta_report = macro_ap_report(dev, meta_oof, strict_official=True)
    meta_ap = float(meta_report["macro_average_precision"])

    fold_reports = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        frame = dev.loc[mask].reset_index(drop=True)
        equal_fold_ap = float(macro_ap_report(frame, equal_six[mask])["macro_average_precision"])
        meta_fold_ap = float(macro_ap_report(frame, meta_oof[mask])["macro_average_precision"])
        weights = np.asarray(crossfit["fold_weights"][int(fold)], dtype=np.float64)
        fold_reports.append(
            {
                "fold": int(fold),
                "rows": int(mask.sum()),
                "equal_six_macro_average_precision": equal_fold_ap,
                "meta_macro_average_precision": meta_fold_ap,
                "delta_vs_equal_six": float(meta_fold_ap - equal_fold_ap),
                "weights": {name: float(weights[i]) for i, name in enumerate(SIX_SIGNAL_NAMES)},
            }
        )

    full_fit_weights = fit_simplex_weights(
        crossfit["rank_matrix"],
        dev["target"].to_numpy(dtype=np.int8),
        dev["category"].to_numpy(dtype=str),
        step_schedule=DEFAULT_STEP_SCHEDULE,
        max_passes=6,
    )
    full_fit_score = np.asarray(crossfit["rank_matrix"], dtype=np.float64) @ full_fit_weights
    full_fit_report = macro_ap_report(dev, full_fit_score, strict_official=True)
    full_fit_ap = float(full_fit_report["macro_average_precision"])

    elapsed_total = time.monotonic() - started
    payload = {
        "version": "v5-strict-global-simplex-meta-blend",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "target_fitted_blender": True,
        "fully_outer_cross_fitted": True,
        "signal_names": list(SIX_SIGNAL_NAMES),
        "equal_six_anchor_macro_ap": equal_six_ap,
        "equal_six_anchor_per_category_ap": equal_six_report["per_category_ap"],
        "strict_crossfit_global_macro_ap": meta_ap,
        "strict_crossfit_global_per_category_ap": meta_report["per_category_ap"],
        "delta_vs_equal_six_anchor": float(meta_ap - equal_six_ap),
        "fold_reports": fold_reports,
        "full_fit_macro_ap_diagnostic_only": full_fit_ap,
        "full_fit_weights": {
            name: float(full_fit_weights[i]) for i, name in enumerate(SIX_SIGNAL_NAMES)
        },
        "step_schedule": [float(v) for v in DEFAULT_STEP_SCHEDULE],
        "max_passes": 6,
        "target_0_60_reached": bool(meta_ap >= 0.60),
        "elapsed_seconds": float(elapsed_total),
        "peak_ram_gib": _peak_ram_gib(),
    }

    metrics_path = output_dir / "v5-meta-blend-metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": folds,
            "equal_six_score": equal_six,
            "meta_oof_score": meta_oof,
        }
    ).to_parquet(output_dir / "v5-meta-blend-oof.parquet", index=False)
    (output_dir / "v5-meta-blend-timing.json").write_text(
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
    run_v5_meta_blend(
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
