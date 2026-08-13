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
from .v5_hgb_stack import DEFAULT_HGB_PARAMS, fit_fixed_hgb_full
from .v5_meta_blend import SIX_SIGNAL_NAMES
from .v5_validation import manifest_sha256
from .v6_fast_ablation import build_fast_candidate_scores


CANDIDATE = "no_teacher"
PRIOR_STRENGTH = 8000.0
STEP_SCHEDULE = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0)
MIN_RAW_OOF = 0.592
MIN_GRAPH_OOF = 0.595


def _peak_ram_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def fit_v10_faststack_production(
    *,
    raw_oof_macro_ap: float,
    graph_oof_macro_ap: float,
    target_stress_graph_mean: float,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    anchor_oof_path: Path,
    typed_fusion_oof_path: Path,
    category_output_path: Path,
    hgb_output_path: Path,
    metadata_output_path: Path,
    expected_split_sha: str,
) -> dict:
    if not np.isfinite(raw_oof_macro_ap) or raw_oof_macro_ap < MIN_RAW_OOF:
        raise ValueError(f"raw_oof_macro_ap must be >= {MIN_RAW_OOF}")
    if not np.isfinite(graph_oof_macro_ap) or graph_oof_macro_ap < MIN_GRAPH_OOF:
        raise ValueError(f"graph_oof_macro_ap must be >= {MIN_GRAPH_OOF}")
    if not np.isfinite(target_stress_graph_mean):
        raise ValueError("target_stress_graph_mean must be finite")
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
        source_name="v10-faststack-production-anchor",
    )
    typed = align_oof_frame(
        [typed_fusion_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("typed_explicit_score",),
        source_name="v10-faststack-production-typed",
    )

    dev = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    right_category = dev["id2"].map(category_by_id)
    if dev["category"].isna().any() or right_category.isna().any():
        raise RuntimeError("failed to attach official categories")
    if not np.array_equal(
        dev["category"].astype(str).to_numpy(), right_category.astype(str).to_numpy()
    ):
        raise RuntimeError("development pair endpoints disagree on category")

    six = {
        **{
            name: anchor[column].to_numpy(dtype=np.float64)
            for name, column in CURRENT5_COLUMNS.items()
        },
        "typed_explicit": typed["typed_explicit_score"].to_numpy(dtype=np.float64),
    }
    candidate_scores = build_fast_candidate_scores(six, CANDIDATE)
    target = dev["target"].to_numpy(dtype=np.int8)
    categories = dev["category"].astype(str).to_numpy()

    category_fit = fit_category_shrunk_full(
        candidate_scores,
        target,
        categories,
        prior_strength=PRIOR_STRENGTH,
        step_schedule=STEP_SCHEDULE,
        max_passes=4,
    )
    category_payload = {
        "version": "v10-faststack-category-shrunk-production",
        "candidate": CANDIDATE,
        "signal_names": list(SIX_SIGNAL_NAMES),
        "global_weights": [float(value) for value in category_fit["global_weights"]],
        "category_weights": {
            str(category): [float(value) for value in weights]
            for category, weights in category_fit["category_weights"].items()
        },
        "category_support": {
            str(category): int(support)
            for category, support in category_fit["category_support"].items()
        },
        "prior_strength": float(PRIOR_STRENGTH),
        "step_schedule": [float(value) for value in STEP_SCHEDULE],
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

    hgb_fit = fit_fixed_hgb_full(candidate_scores, target, categories)
    hgb_bundle = {
        "version": "v10-faststack-hgb-production",
        "candidate": CANDIDATE,
        "signal_names": list(SIX_SIGNAL_NAMES),
        "category_names": list(hgb_fit["category_names"]),
        "model": hgb_fit["model"],
        "params": dict(DEFAULT_HGB_PARAMS),
        "sklearn_version": str(sklearn.__version__),
        "split_sha256": expected_split_sha,
        "production_refit_uses_all_development_labels": True,
        "production_refit_score_is_not_validation": True,
    }
    hgb_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hgb_bundle, hgb_output_path, compress=3)

    payload = {
        "version": "v10-faststack-production-refit",
        "candidate": CANDIDATE,
        "strict_raw_oof_macro_ap": float(raw_oof_macro_ap),
        "strict_graph_oof_macro_ap": float(graph_oof_macro_ap),
        "target_stress_graph_mean": float(target_stress_graph_mean),
        "graph_config": {
            "reciprocal_best_bonus": 0.0,
            "reciprocal_top3_bonus": 0.0,
            "endpoint_rank_weight": 0.02,
            "ambiguity_penalty": 0.01,
        },
        "split_sha256": expected_split_sha,
        "selection_gold_metric_opened": False,
        "selection_gold_rows_scored": 0,
        "production_refit_uses_all_development_labels": True,
        "production_refit_score_is_not_validation": True,
        "teacher_checkpoint_required": False,
        "required_expensive_signals": ["contrastive"],
        "teacher_surrogate": "unweighted_mean_target_free_percentile_ranks_of_weak_sparse_explicit_contrastive_typed_explicit",
        "fusion_formula": "0.5*percentile_rank(category_shrunk_score)+0.5*percentile_rank(hgb_score), then frozen target-free graph rescore",
        "development_rows": int(len(dev)),
        "sklearn_version": str(sklearn.__version__),
        "python_version": platform.python_version(),
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_ram_gib": _peak_ram_gib(),
    }
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-oof-macro-ap", type=float, required=True)
    p.add_argument("--graph-oof-macro-ap", type=float, required=True)
    p.add_argument("--target-stress-graph-mean", type=float, required=True)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--anchor-oof", type=Path, required=True)
    p.add_argument("--typed-fusion-oof", type=Path, required=True)
    p.add_argument("--category-output", type=Path, required=True)
    p.add_argument("--hgb-output", type=Path, required=True)
    p.add_argument("--metadata-output", type=Path, required=True)
    p.add_argument("--expected-split-sha", required=True)
    a = p.parse_args()
    fit_v10_faststack_production(
        raw_oof_macro_ap=a.raw_oof_macro_ap,
        graph_oof_macro_ap=a.graph_oof_macro_ap,
        target_stress_graph_mean=a.target_stress_graph_mean,
        items_path=a.items,
        matches_path=a.matches,
        manifest_path=a.manifest,
        anchor_oof_path=a.anchor_oof,
        typed_fusion_oof_path=a.typed_fusion_oof,
        category_output_path=a.category_output,
        hgb_output_path=a.hgb_output,
        metadata_output_path=a.metadata_output,
        expected_split_sha=a.expected_split_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
