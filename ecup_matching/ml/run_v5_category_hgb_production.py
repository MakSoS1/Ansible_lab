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


STRICT_SELECTED_OOF_MACRO_AP = 0.6018115534135564
CATEGORY_COMPONENT_OOF_MACRO_AP = 0.60095424180184
HGB_COMPONENT_OOF_MACRO_AP = 0.6006290884983169
PRIOR_STRENGTH = 8000.0
STEP_SCHEDULE = (1.0 / 12.0, 1.0 / 24.0, 1.0 / 48.0)
MAX_PASSES = 4


def _peak_ram_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _float_list(values) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).tolist()]


def fit_production_bundle(
    *,
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
        raise RuntimeError("failed to attach official category to every development pair")
    observed_categories = set(dev["category"].astype(str).unique().tolist())
    expected_categories = set(OFFICIAL_CATEGORIES)
    if observed_categories != expected_categories:
        raise RuntimeError(
            f"development category set mismatch; missing={sorted(expected_categories-observed_categories)}, "
            f"extra={sorted(observed_categories-expected_categories)}"
        )

    current5 = {
        name: anchor[column].to_numpy(dtype=np.float64)
        for name, column in CURRENT5_COLUMNS.items()
    }
    six_scores = {
        **current5,
        "typed_explicit": typed["typed_explicit_score"].to_numpy(dtype=np.float64),
    }
    if tuple(six_scores) != SIX_SIGNAL_NAMES:
        raise RuntimeError(f"six-signal order mismatch: {tuple(six_scores)} != {SIX_SIGNAL_NAMES}")

    target = dev["target"].to_numpy(dtype=np.int8)
    categories = dev["category"].to_numpy(dtype=str)

    print(
        f"[category-hgb-production] phase=fit_category rows={len(dev)} categories={len(observed_categories)} "
        f"elapsed={time.monotonic()-started:.1f}s peak_ram_gib={_peak_ram_gib():.3f}",
        flush=True,
    )
    category_fit = fit_category_shrunk_full(
        six_scores,
        target,
        categories,
        prior_strength=PRIOR_STRENGTH,
        step_schedule=STEP_SCHEDULE,
        max_passes=MAX_PASSES,
    )

    print(
        f"[category-hgb-production] phase=fit_hgb rows={len(dev)} "
        f"elapsed={time.monotonic()-started:.1f}s peak_ram_gib={_peak_ram_gib():.3f}",
        flush=True,
    )
    hgb_fit = fit_fixed_hgb_full(six_scores, target, categories)

    category_payload = {
        "version": "v5-category-shrunk-production-refit-for-category-hgb-fusion",
        "signal_names": list(SIX_SIGNAL_NAMES),
        "prior_strength": PRIOR_STRENGTH,
        "step_schedule": [float(value) for value in STEP_SCHEDULE],
        "max_passes": MAX_PASSES,
        "global_weights": _float_list(category_fit["global_weights"]),
        "category_weights": {
            category: _float_list(category_fit["category_weights"][category])
            for category in OFFICIAL_CATEGORIES
        },
        "category_support": {
            category: int(category_fit["category_support"][category])
            for category in OFFICIAL_CATEGORIES
        },
    }
    for category, weights in category_payload["category_weights"].items():
        array = np.asarray(weights, dtype=np.float64)
        if np.any(array < -1e-12) or not np.isclose(array.sum(), 1.0, rtol=0.0, atol=1e-8):
            raise RuntimeError(f"invalid production simplex for category {category!r}")

    hgb_bundle = {
        "version": "v5-fixed-hgb-production-refit-for-category-hgb-fusion",
        "signal_names": list(SIX_SIGNAL_NAMES),
        "category_names": list(hgb_fit["category_names"]),
        "params": dict(hgb_fit["params"]),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "model": hgb_fit["model"],
    }
    if tuple(hgb_bundle["category_names"]) != tuple(sorted(OFFICIAL_CATEGORIES)):
        raise RuntimeError("production HGB category vocabulary mismatch")

    elapsed = float(time.monotonic() - started)
    metadata = {
        "version": "v5-category-shrunk-hgb-production-refit",
        "selection_method": "fixed 50/50 percentile-rank fusion of two fully outer-cross-fitted components",
        "strict_selected_oof_macro_ap": STRICT_SELECTED_OOF_MACRO_AP,
        "category_component_oof_macro_ap": CATEGORY_COMPONENT_OOF_MACRO_AP,
        "hgb_component_oof_macro_ap": HGB_COMPONENT_OOF_MACRO_AP,
        "selection_gold_metric_opened": False,
        "selection_gold_rows_scored": 0,
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "signal_names": list(SIX_SIGNAL_NAMES),
        "fusion_formula": "0.5*percentile_rank(category_shrunk_score)+0.5*percentile_rank(hgb_score)",
        "production_refit_uses_all_development_labels": True,
        "production_refit_score_is_not_validation": True,
        "category_prior_strength": PRIOR_STRENGTH,
        "hgb_params": dict(DEFAULT_HGB_PARAMS),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "elapsed_seconds": elapsed,
        "peak_ram_gib": _peak_ram_gib(),
    }

    category_output_path.parent.mkdir(parents=True, exist_ok=True)
    hgb_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    category_output_path.write_text(
        json.dumps(category_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    joblib.dump(hgb_bundle, hgb_output_path, compress=3)
    metadata_output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    loaded = joblib.load(hgb_output_path)
    if tuple(loaded["signal_names"]) != SIX_SIGNAL_NAMES:
        raise RuntimeError("round-trip HGB bundle signal order mismatch")
    if tuple(loaded["category_names"]) != tuple(sorted(OFFICIAL_CATEGORIES)):
        raise RuntimeError("round-trip HGB bundle category vocabulary mismatch")
    if not hasattr(loaded["model"], "predict_proba"):
        raise RuntimeError("round-trip HGB bundle is missing predict_proba model")

    print(
        f"[category-hgb-production] phase=done elapsed={elapsed:.1f}s peak_ram_gib={metadata['peak_ram_gib']:.3f} "
        f"category_json={category_output_path} hgb_joblib={hgb_output_path}",
        flush=True,
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True), flush=True)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
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
    fit_production_bundle(
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
