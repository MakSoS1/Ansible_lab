from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data_subset import select_items_by_ids
from .run_v5_fixed_blend import align_oof_frame
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v5_typed_explicit_fusion import CURRENT5_COLUMNS
from .v5_category_shrunk import fit_category_shrunk_full
from .v5_evaluation import OFFICIAL_CATEGORIES
from .v5_hgb_stack import DEFAULT_HGB_PARAMS, fit_fixed_hgb_full
from .v5_meta_blend import SIX_SIGNAL_NAMES, _rank_signals
from .v5_validation import manifest_sha256
from .v6_teacher_gate import disagreement_gate_mask
from .v6_teacher_hybrid import HYBRID_COVERAGES, empirical_cdf_from_train
from .v6_teacher_student import (
    NON_TEACHER_SIGNAL_NAMES,
    fit_teacher_student_full,
    predict_teacher_student,
)


PRIOR_STRENGTH = 8000.0
STEP_SCHEDULE = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0)


def _peak_ram_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def fit_v6_hybrid_production(
    *,
    coverage: float,
    selected_oof_macro_ap: float,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    anchor_oof_path: Path,
    typed_fusion_oof_path: Path,
    student_output_path: Path,
    teacher_reference_output_path: Path,
    category_output_path: Path,
    hgb_output_path: Path,
    metadata_output_path: Path,
    expected_split_sha: str,
) -> dict:
    if coverage not in HYBRID_COVERAGES:
        raise ValueError(f"coverage must be one of {HYBRID_COVERAGES}")
    if not np.isfinite(selected_oof_macro_ap) or selected_oof_macro_ap < 0.60:
        raise ValueError("selected_oof_macro_ap must be finite and >= 0.60")
    started = time.monotonic()

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
        source_name="v6-hybrid-production-anchor",
    )
    typed = align_oof_frame(
        [typed_fusion_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("typed_explicit_score",),
        source_name="v6-hybrid-production-typed",
    )

    dev = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach official categories")
    categories = dev["category"].astype(str).to_numpy()
    target = dev["target"].to_numpy(dtype=np.int8)

    six = {
        **{
            name: anchor[column].to_numpy(dtype=np.float64)
            for name, column in CURRENT5_COLUMNS.items()
        },
        "typed_explicit": typed["typed_explicit_score"].to_numpy(dtype=np.float64),
    }
    non_teacher = {name: six[name] for name in NON_TEACHER_SIGNAL_NAMES}
    teacher = np.asarray(six["teacher"], dtype=np.float64)

    student_bundle = fit_teacher_student_full(non_teacher, teacher, categories)
    student_score = predict_teacher_student(student_bundle, non_teacher, categories)
    selected = disagreement_gate_mask(non_teacher, categories, coverage=coverage)
    hybrid_teacher = student_score.copy()
    if selected.any():
        hybrid_teacher[selected] = empirical_cdf_from_train(teacher, teacher[selected])
    hybrid_scores = {**non_teacher, "teacher": hybrid_teacher}

    student_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": "v6-hybrid-teacher-student-production",
            "model": student_bundle["model"],
            "category_names": list(student_bundle["category_names"]),
            "params": dict(student_bundle["params"]),
            "non_teacher_signal_names": list(NON_TEACHER_SIGNAL_NAMES),
            "split_sha256": expected_split_sha,
        },
        student_output_path,
        compress=3,
    )
    teacher_reference_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        teacher_reference_output_path,
        np.sort(teacher.astype(np.float32), kind="mergesort"),
        allow_pickle=False,
    )

    ranked = _rank_signals(hybrid_scores, SIX_SIGNAL_NAMES)
    category_fit = fit_category_shrunk_full(
        ranked,
        target,
        categories,
        signal_names=SIX_SIGNAL_NAMES,
        prior_strength=PRIOR_STRENGTH,
        category_names=OFFICIAL_CATEGORIES,
        step_schedule=STEP_SCHEDULE,
        max_passes=4,
    )
    category_payload = {
        "version": "v6-hybrid-category-shrunk-production",
        "coverage": float(coverage),
        "signal_names": list(SIX_SIGNAL_NAMES),
        "global_weights": [float(v) for v in category_fit["global_weights"]],
        "category_weights": category_fit["category_weights"],
        "category_support": category_fit["category_support"],
        "prior_strength": float(PRIOR_STRENGTH),
        "step_schedule": [float(v) for v in STEP_SCHEDULE],
        "max_passes": 4,
        "split_sha256": expected_split_sha,
        "production_refit_uses_all_development_labels": True,
        "production_refit_score_is_not_validation": True,
    }
    category_output_path.parent.mkdir(parents=True, exist_ok=True)
    category_output_path.write_text(
        json.dumps(category_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    hgb_fit = fit_fixed_hgb_full(hybrid_scores, target, categories)
    hgb_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": "v6-hybrid-hgb-production",
            "coverage": float(coverage),
            "signal_names": list(SIX_SIGNAL_NAMES),
            "category_names": list(hgb_fit["category_names"]),
            "model": hgb_fit["model"],
            "params": dict(DEFAULT_HGB_PARAMS),
            "sklearn_version": str(sklearn.__version__),
            "split_sha256": expected_split_sha,
            "production_refit_uses_all_development_labels": True,
            "production_refit_score_is_not_validation": True,
        },
        hgb_output_path,
        compress=3,
    )

    payload = {
        "version": "v6-hybrid-production-refit",
        "coverage": float(coverage),
        "production_gate_fraction_on_development": float(np.mean(selected)),
        "production_gate_rows_on_development": int(selected.sum()),
        "strict_selected_oof_macro_ap": float(selected_oof_macro_ap),
        "quality_gate_macro_ap": 0.60,
        "architecture": "four structured + full contrastive + HGB teacher student + real teacher only on target-free disagreement gate",
        "teacher_reference": "sorted full-development outer-fold teacher sigmoid scores",
        "teacher_reference_rows": int(len(teacher)),
        "split_sha256": expected_split_sha,
        "selection_gold_metric_opened": False,
        "selection_gold_rows_scored": 0,
        "production_refit_uses_all_development_labels": True,
        "production_refit_score_is_not_validation": True,
        "fusion_formula": "0.5*percentile_rank(category_shrunk_score)+0.5*percentile_rank(hgb_score)",
        "development_rows": int(len(dev)),
        "sklearn_version": str(sklearn.__version__),
        "python_version": platform.python_version(),
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_ram_gib": _peak_ram_gib(),
    }
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--coverage", type=float, required=True)
    p.add_argument("--selected-oof-macro-ap", type=float, required=True)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--anchor-oof", type=Path, required=True)
    p.add_argument("--typed-fusion-oof", type=Path, required=True)
    p.add_argument("--student-output", type=Path, required=True)
    p.add_argument("--teacher-reference-output", type=Path, required=True)
    p.add_argument("--category-output", type=Path, required=True)
    p.add_argument("--hgb-output", type=Path, required=True)
    p.add_argument("--metadata-output", type=Path, required=True)
    p.add_argument("--expected-split-sha", required=True)
    a = p.parse_args()
    fit_v6_hybrid_production(
        coverage=a.coverage,
        selected_oof_macro_ap=a.selected_oof_macro_ap,
        items_path=a.items,
        matches_path=a.matches,
        manifest_path=a.manifest,
        anchor_oof_path=a.anchor_oof,
        typed_fusion_oof_path=a.typed_fusion_oof,
        student_output_path=a.student_output,
        teacher_reference_output_path=a.teacher_reference_output,
        category_output_path=a.category_output,
        hgb_output_path=a.hgb_output,
        metadata_output_path=a.metadata_output,
        expected_split_sha=a.expected_split_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
