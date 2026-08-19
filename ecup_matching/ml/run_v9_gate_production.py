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
from .metrics import OFFICIAL_CATEGORIES
from .run_v5_fixed_blend import align_oof_frame
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .run_v5_typed_explicit_fusion import CURRENT5_COLUMNS
from .v5_category_shrunk import fit_category_shrunk_full
from .v5_hgb_stack import DEFAULT_HGB_PARAMS, fit_fixed_hgb_full
from .v5_meta_blend import SIX_SIGNAL_NAMES
from .v5_validation import manifest_sha256
from .v6_teacher_gate import GATE_COVERAGES, build_teacher_gated_scores


PRIOR_STRENGTH = 8000.0
STEP_SCHEDULE = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0)
MAX_PASSES = 4
EXPECTED_COVERAGE = 0.40
EXPECTED_BASE_OOF = 0.595505427416499
EXPECTED_GRAPH_OOF = 0.597005931143384
EXPECTED_TARGET_STRESS = 0.4515676235464289
EXPECTED_SPLIT_SHA = "aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b"
GRAPH_CONFIG = {"rb": 0.0, "rt": 0.0, "ep": 0.02, "ap": 0.01}
METRIC_ATOL = 1e-12


def _peak_ram_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _float_list(values) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).tolist()]


def fit_v6_gate_production(
    *,
    coverage: float,
    selected_oof_macro_ap: float,
    quality_gate_macro_ap: float,
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
    """Isolated v9 copy of the v6 full-refit routine with an explicit frozen gate.

    The v6/v8 implementation intentionally hard-codes >=0.60.  v9 keeps that
    historical contract untouched and instead uses this isolated implementation
    whose accepted metric is the exact pre-selected gate40 fold-local graph OOF.
    """
    coverage = float(coverage)
    selected_oof_macro_ap = float(selected_oof_macro_ap)
    quality_gate_macro_ap = float(quality_gate_macro_ap)
    if coverage not in GATE_COVERAGES:
        raise ValueError(f"coverage must be one of {GATE_COVERAGES}")
    if not np.isfinite(quality_gate_macro_ap) or quality_gate_macro_ap <= 0.0:
        raise ValueError("quality_gate_macro_ap must be finite and positive")
    if not np.isfinite(selected_oof_macro_ap) or selected_oof_macro_ap < quality_gate_macro_ap:
        raise ValueError(
            f"selected_oof_macro_ap must be finite and >= frozen v9 quality gate {quality_gate_macro_ap}"
        )
    started = time.monotonic()

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
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
        source_name="v9-gate-production-anchor",
    )
    typed = align_oof_frame(
        [typed_fusion_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("typed_explicit_score",),
        source_name="v9-gate-production-typed",
    )

    dev = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach official categories")
    observed_categories = set(dev["category"].astype(str).unique().tolist())
    expected_categories = set(OFFICIAL_CATEGORIES)
    if observed_categories != expected_categories:
        raise RuntimeError(
            f"development category set mismatch; missing={sorted(expected_categories-observed_categories)}, "
            f"extra={sorted(observed_categories-expected_categories)}"
        )

    six = {
        **{
            name: anchor[column].to_numpy(dtype=np.float64)
            for name, column in CURRENT5_COLUMNS.items()
        },
        "typed_explicit": typed["typed_explicit_score"].to_numpy(dtype=np.float64),
    }
    if tuple(six) != SIX_SIGNAL_NAMES:
        raise RuntimeError(f"six-signal order mismatch: {tuple(six)} != {SIX_SIGNAL_NAMES}")
    categories = dev["category"].astype(str).to_numpy()
    target = dev["target"].to_numpy(dtype=np.int8)
    gated, teacher_mask = build_teacher_gated_scores(six, categories, coverage=coverage)
    actual_fraction = float(np.mean(teacher_mask))

    print(
        f"[v9-gate-production] phase=fit_category rows={len(dev)} coverage={coverage:.2f} "
        f"teacher_fraction={actual_fraction:.6f} elapsed={time.monotonic()-started:.1f}s",
        flush=True,
    )
    category_fit = fit_category_shrunk_full(
        gated,
        target,
        categories,
        prior_strength=PRIOR_STRENGTH,
        step_schedule=STEP_SCHEDULE,
        max_passes=MAX_PASSES,
    )
    category_payload = {
        "version": "v9-gate-category-shrunk-production",
        "coverage": coverage,
        "signal_names": list(SIX_SIGNAL_NAMES),
        "global_weights": _float_list(category_fit["global_weights"]),
        "category_weights": {
            category: _float_list(category_fit["category_weights"][category])
            for category in OFFICIAL_CATEGORIES
        },
        "category_support": {
            category: int(category_fit["category_support"][category])
            for category in OFFICIAL_CATEGORIES
        },
        "prior_strength": float(PRIOR_STRENGTH),
        "step_schedule": [float(value) for value in STEP_SCHEDULE],
        "max_passes": MAX_PASSES,
        "split_sha256": expected_split_sha,
        "production_refit_uses_all_development_labels": True,
        "production_refit_score_is_not_validation": True,
    }
    for category, weights in category_payload["category_weights"].items():
        array = np.asarray(weights, dtype=np.float64)
        if np.any(array < -1e-12) or not np.isclose(array.sum(), 1.0, rtol=0.0, atol=1e-8):
            raise RuntimeError(f"invalid production simplex for category {category!r}")
    Path(category_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(category_output_path).write_text(
        json.dumps(category_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        f"[v9-gate-production] phase=fit_hgb rows={len(dev)} "
        f"elapsed={time.monotonic()-started:.1f}s peak_ram_gib={_peak_ram_gib():.3f}",
        flush=True,
    )
    hgb_fit = fit_fixed_hgb_full(gated, target, categories)
    hgb_bundle = {
        "version": "v9-gate-hgb-production",
        "coverage": coverage,
        "signal_names": list(SIX_SIGNAL_NAMES),
        "category_names": list(hgb_fit["category_names"]),
        "model": hgb_fit["model"],
        "params": dict(hgb_fit["params"]),
        "sklearn_version": str(sklearn.__version__),
        "python_version": platform.python_version(),
        "split_sha256": expected_split_sha,
        "production_refit_uses_all_development_labels": True,
        "production_refit_score_is_not_validation": True,
    }
    if tuple(hgb_bundle["category_names"]) != tuple(sorted(OFFICIAL_CATEGORIES)):
        raise RuntimeError("production HGB category vocabulary mismatch")
    Path(hgb_output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hgb_bundle, hgb_output_path, compress=3)

    elapsed = float(time.monotonic() - started)
    payload = {
        "version": "v9-base-gate-production-refit",
        "coverage": coverage,
        "actual_teacher_fraction_on_development": actual_fraction,
        "teacher_rows_on_development": int(teacher_mask.sum()),
        "strict_selected_oof_macro_ap": selected_oof_macro_ap,
        "quality_gate_macro_ap": quality_gate_macro_ap,
        "architecture": "four structured + contrastive + real teacher on target-free disagreement gate; target-free surrogate otherwise",
        "teacher_selected_transform": "percentile rank over selected teacher scores only",
        "unselected_teacher_surrogate": "mean percentile rank of weak+sparse+explicit+contrastive+typed_explicit",
        "split_sha256": expected_split_sha,
        "selection_gold_metric_opened": False,
        "selection_gold_rows_scored": 0,
        "production_refit_uses_all_development_labels": True,
        "production_refit_score_is_not_validation": True,
        "fusion_formula": "0.5*percentile_rank(category_shrunk_score)+0.5*percentile_rank(hgb_score)",
        "development_rows": int(len(dev)),
        "category_prior_strength": PRIOR_STRENGTH,
        "hgb_params": dict(DEFAULT_HGB_PARAMS),
        "sklearn_version": str(sklearn.__version__),
        "python_version": platform.python_version(),
        "elapsed_seconds": elapsed,
        "peak_ram_gib": _peak_ram_gib(),
    }
    Path(metadata_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    loaded = joblib.load(hgb_output_path)
    if tuple(loaded["signal_names"]) != SIX_SIGNAL_NAMES:
        raise RuntimeError("round-trip HGB bundle signal order mismatch")
    if tuple(loaded["category_names"]) != tuple(sorted(OFFICIAL_CATEGORIES)):
        raise RuntimeError("round-trip HGB bundle category vocabulary mismatch")
    if not hasattr(loaded["model"], "predict_proba"):
        raise RuntimeError("round-trip HGB bundle is missing predict_proba model")

    print(
        f"[v9-gate-production] phase=done elapsed={elapsed:.1f}s peak_ram_gib={payload['peak_ram_gib']:.3f}",
        flush=True,
    )
    return payload


def fit_v9_gate40_production(
    *,
    coverage: float,
    base_oof_macro_ap: float,
    graph_oof_macro_ap: float,
    target_stress_mean: float,
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
    coverage = float(coverage)
    base_oof_macro_ap = float(base_oof_macro_ap)
    graph_oof_macro_ap = float(graph_oof_macro_ap)
    target_stress_mean = float(target_stress_mean)
    checks = (
        (abs(coverage - EXPECTED_COVERAGE) <= METRIC_ATOL, "coverage"),
        (abs(base_oof_macro_ap - EXPECTED_BASE_OOF) <= METRIC_ATOL, "frozen base OOF"),
        (abs(graph_oof_macro_ap - EXPECTED_GRAPH_OOF) <= METRIC_ATOL, "frozen graph OOF"),
        (abs(target_stress_mean - EXPECTED_TARGET_STRESS) <= METRIC_ATOL, "frozen target stress"),
        (str(expected_split_sha) == EXPECTED_SPLIT_SHA, "split SHA"),
    )
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise ValueError(f"invalid v9 gate40 frozen evidence: {failed}")

    payload = fit_v6_gate_production(
        coverage=coverage,
        selected_oof_macro_ap=graph_oof_macro_ap,
        quality_gate_macro_ap=EXPECTED_GRAPH_OOF,
        items_path=Path(items_path),
        matches_path=Path(matches_path),
        manifest_path=Path(manifest_path),
        anchor_oof_path=Path(anchor_oof_path),
        typed_fusion_oof_path=Path(typed_fusion_oof_path),
        category_output_path=Path(category_output_path),
        hgb_output_path=Path(hgb_output_path),
        metadata_output_path=Path(metadata_output_path),
        expected_split_sha=str(expected_split_sha),
    )
    if payload.get("selection_gold_metric_opened") is not False or int(payload.get("selection_gold_rows_scored", -1)) != 0:
        raise RuntimeError("v9 base production refit violated sealed-gold contract")

    result = dict(payload)
    result.update(
        {
            "version": "v9-gate40-production-refit",
            "base_strict_oof_macro_ap": base_oof_macro_ap,
            "fold_local_graph_strict_oof_macro_ap": graph_oof_macro_ap,
            "target_stress_mean": target_stress_mean,
            "target_stress_ratio": 0.566880890615799,
            "graph_config": dict(GRAPH_CONFIG),
            "selection_basis": "frozen gate40 Pareto: strict OOF + fold-local graph + target-stress + runtime veto",
            "leaderboard_anchor_v7_observed_by_owner": 0.36,
            "leaderboard_anchor_used_for_fitting": False,
            "selection_gold_metric_opened": False,
            "selection_gold_rows_scored": 0,
        }
    )
    Path(metadata_output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=float, required=True)
    parser.add_argument("--base-oof-macro-ap", type=float, required=True)
    parser.add_argument("--graph-oof-macro-ap", type=float, required=True)
    parser.add_argument("--target-stress-mean", type=float, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-oof", type=Path, required=True)
    parser.add_argument("--typed-fusion-oof", type=Path, required=True)
    parser.add_argument("--category-output", type=Path, required=True)
    parser.add_argument("--hgb-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    args = parser.parse_args()
    fit_v9_gate40_production(
        coverage=args.coverage,
        base_oof_macro_ap=args.base_oof_macro_ap,
        graph_oof_macro_ap=args.graph_oof_macro_ap,
        target_stress_mean=args.target_stress_mean,
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        anchor_oof_path=args.anchor_oof,
        typed_fusion_oof_path=args.typed_fusion_oof,
        category_output_path=args.category_output,
        hgb_output_path=args.hgb_output,
        metadata_output_path=args.metadata_output,
        expected_split_sha=args.expected_split_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
