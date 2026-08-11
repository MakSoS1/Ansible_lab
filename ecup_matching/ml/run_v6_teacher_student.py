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
from .v6_teacher_student import NON_TEACHER_SIGNAL_NAMES, crossfit_teacher_student


PRIOR_STRENGTH = 8000.0
STEP_SCHEDULE = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0)


def _peak_ram_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def run_teacher_student(
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
    if manifest_sha256(manifest) != expected_split_sha:
        raise ValueError("sealed split SHA mismatch")

    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, folds = development_rows_and_folds(manifest, total_rows=len(matches))
    dev_rows = np.asarray(dev_rows, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int16)
    anchor = align_oof_frame(
        [anchor_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=tuple(CURRENT5_COLUMNS.values()),
        source_name="v6-teacher-student-anchor",
    )
    typed = align_oof_frame(
        [typed_fusion_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("typed_explicit_score",),
        source_name="v6-teacher-student-typed",
    )
    dev = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach official categories")
    categories = dev["category"].astype(str).to_numpy()

    six = {
        **{
            name: anchor[column].to_numpy(dtype=np.float64)
            for name, column in CURRENT5_COLUMNS.items()
        },
        "typed_explicit": typed["typed_explicit_score"].to_numpy(dtype=np.float64),
    }
    non_teacher = {name: six[name] for name in NON_TEACHER_SIGNAL_NAMES}
    student = crossfit_teacher_student(
        non_teacher, six["teacher"], categories, folds
    )
    student_teacher = np.asarray(student["oof_score"], dtype=np.float64)
    candidate_scores = {**non_teacher, "teacher": student_teacher}
    target = dev["target"].to_numpy(dtype=np.int8)
    print(
        f"[v6-student] phase=student rows={len(dev)} elapsed={time.monotonic()-started:.1f}s "
        f"peak_ram_gib={_peak_ram_gib():.3f}",
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
            f"[v6-student] phase=category fold={fold} done={done}/{total} elapsed={time.monotonic()-started:.1f}s",
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
            f"[v6-student] phase=hgb fold={fold} done={done}/{total} elapsed={time.monotonic()-started:.1f}s",
            flush=True,
        ),
        **hgb_params,
    )
    hgb_oof = np.asarray(hgb_result["oof_score"], dtype=np.float64)
    hgb_report = macro_ap_report(dev, hgb_oof, strict_official=True)
    fusion = 0.5 * percentile_rank(category_oof) + 0.5 * percentile_rank(hgb_oof)
    fusion_report = macro_ap_report(dev, fusion, strict_official=True)
    fusion_ap = float(fusion_report["macro_average_precision"])

    payload = {
        "version": "v6-teacher-student-v1",
        "architecture": "weak+sparse+explicit+contrastive+typed_explicit + HGB-distilled teacher",
        "student_target": "teacher percentile rank fitted separately on each outer-train partition",
        "student_held_teacher_values_used": False,
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "fully_outer_cross_fitted_student": True,
        "fully_outer_cross_fitted_meta": True,
        "post_result_weight_search": False,
        "category_shrunk_macro_ap": float(category_report["macro_average_precision"]),
        "hgb_macro_ap": float(hgb_report["macro_average_precision"]),
        "fusion_formula": "0.5*percentile_rank(category_shrunk)+0.5*percentile_rank(hgb)",
        "fusion_macro_ap": fusion_ap,
        "fusion_per_category_ap": fusion_report["per_category_ap"],
        "target_0_60_reached": bool(fusion_ap >= 0.60),
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_ram_gib": _peak_ram_gib(),
    }
    (output_dir / "v6-teacher-student-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": folds,
            "student_teacher_oof_score": student_teacher,
            "category_shrunk_oof_score": category_oof,
            "hgb_stack_oof_score": hgb_oof,
            "fusion_oof_score": fusion,
        }
    ).to_parquet(output_dir / "v6-teacher-student-oof.parquet", index=False)
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
    run_teacher_student(
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
