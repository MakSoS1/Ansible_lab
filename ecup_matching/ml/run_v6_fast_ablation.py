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
from .v5_category_shrunk import crossfit_category_shrunk_simplex
from .v5_evaluation import macro_ap_report
from .v5_fixed_blend import percentile_rank
from .v5_hgb_stack import DEFAULT_HGB_PARAMS, crossfit_fixed_hgb_stack
from .v5_validation import manifest_sha256
from .v6_fast_ablation import CANDIDATE_SPECS, build_fast_candidate_scores


PRIOR_STRENGTH = 8000.0
STEP_SCHEDULE = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0)


def _peak_ram_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def run_fast_ablation(
    *,
    candidate: str,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    anchor_oof_path: Path,
    typed_fusion_oof_path: Path,
    output_dir: Path,
    expected_split_sha: str,
) -> dict:
    if candidate not in CANDIDATE_SPECS:
        raise ValueError(f"unknown fast candidate: {candidate}")
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
        source_name="v6-fast-anchor",
    )
    typed = align_oof_frame(
        [typed_fusion_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("typed_explicit_score",),
        source_name="v6-fast-typed",
    )

    dev = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach official categories")

    six = {
        **{
            name: anchor[column].to_numpy(dtype=np.float64)
            for name, column in CURRENT5_COLUMNS.items()
        },
        "typed_explicit": typed["typed_explicit_score"].to_numpy(dtype=np.float64),
    }
    candidate_scores = build_fast_candidate_scores(six, candidate)
    target = dev["target"].to_numpy(dtype=np.int8)
    categories = dev["category"].to_numpy(dtype=str)

    print(
        f"[v6-fast] candidate={candidate} phase=loaded rows={len(dev)} "
        f"expensive={CANDIDATE_SPECS[candidate].required_expensive_signals} "
        f"elapsed={time.monotonic()-started:.1f}s peak_ram_gib={_peak_ram_gib():.3f}",
        flush=True,
    )

    category_result = crossfit_category_shrunk_simplex(
        candidate_scores,
        target,
        categories,
        folds,
        prior_strength=PRIOR_STRENGTH,
        step_schedule=STEP_SCHEDULE,
        max_passes=4,
        progress=lambda done, total, fold: print(
            f"[v6-fast] candidate={candidate} phase=category fold={fold} done={done}/{total} "
            f"elapsed={time.monotonic()-started:.1f}s",
            flush=True,
        ),
    )
    category_oof = np.asarray(category_result["oof_score"], dtype=np.float64)
    category_report = macro_ap_report(dev, category_oof, strict_official=True)

    hgb_params = {k: v for k, v in DEFAULT_HGB_PARAMS.items() if k != "early_stopping"}
    hgb_result = crossfit_fixed_hgb_stack(
        candidate_scores,
        target,
        categories,
        folds,
        progress=lambda done, total, fold: print(
            f"[v6-fast] candidate={candidate} phase=hgb fold={fold} done={done}/{total} "
            f"elapsed={time.monotonic()-started:.1f}s",
            flush=True,
        ),
        **hgb_params,
    )
    hgb_oof = np.asarray(hgb_result["oof_score"], dtype=np.float64)
    hgb_report = macro_ap_report(dev, hgb_oof, strict_official=True)

    fusion = 0.5 * percentile_rank(category_oof) + 0.5 * percentile_rank(hgb_oof)
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
                "category_shrunk_macro_ap": float(
                    macro_ap_report(frame, category_oof[mask])["macro_average_precision"]
                ),
                "hgb_macro_ap": float(
                    macro_ap_report(frame, hgb_oof[mask])["macro_average_precision"]
                ),
                "fusion_macro_ap": float(
                    macro_ap_report(frame, fusion[mask])["macro_average_precision"]
                ),
            }
        )

    payload = {
        "version": "v6-fast-ablation-v1",
        "candidate": candidate,
        "candidate_spec": {
            "retained_signals": list(CANDIDATE_SPECS[candidate].retained_signals),
            "required_expensive_signals": list(
                CANDIDATE_SPECS[candidate].required_expensive_signals
            ),
            "surrogate_targets": list(CANDIDATE_SPECS[candidate].surrogate_targets),
            "surrogate": "unweighted_mean_of_target_free_percentile_ranks",
        },
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "fully_outer_cross_fitted_meta": True,
        "candidate_set_frozen_before_metric": True,
        "post_result_weight_search": False,
        "category_shrunk_macro_ap": float(category_report["macro_average_precision"]),
        "hgb_macro_ap": float(hgb_report["macro_average_precision"]),
        "fusion_formula": "0.5*percentile_rank(category_shrunk)+0.5*percentile_rank(hgb)",
        "fusion_macro_ap": fusion_ap,
        "fusion_per_category_ap": fusion_report["per_category_ap"],
        "fold_reports": fold_reports,
        "target_0_60_reached": bool(fusion_ap >= 0.60),
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_ram_gib": _peak_ram_gib(),
    }
    stem = f"v6-fast-{candidate}"
    (output_dir / f"{stem}-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": folds,
            "category_shrunk_oof_score": category_oof,
            "hgb_stack_oof_score": hgb_oof,
            "fusion_oof_score": fusion,
        }
    ).to_parquet(output_dir / f"{stem}-oof.parquet", index=False)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(CANDIDATE_SPECS), required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-oof", type=Path, required=True)
    parser.add_argument("--typed-fusion-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    args = parser.parse_args()
    run_fast_ablation(
        candidate=args.candidate,
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
