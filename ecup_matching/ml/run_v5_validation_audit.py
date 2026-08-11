from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .features_v2 import build_features_v2_chunked
from .train_v1 import attach_pair_category, category_equalizing_weights
from .train_v2_structured import fit_estimator
from .v5_evaluation import macro_ap_report
from .v5_validation import (
    build_v5_split_manifest,
    manifest_sha256,
    validate_manifest_no_overlap,
)


def _bucket(series: pd.Series, bins: list[float]) -> pd.Series:
    labels = [f"b{i}" for i in range(len(bins) - 1)]
    return pd.cut(
        pd.to_numeric(series, errors="coerce").fillna(-1.0),
        bins=bins,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    ).astype("string").fillna("missing")


def build_split_descriptors(pairs: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    required_pairs = {"category", "target"}
    missing_pairs = required_pairs - set(pairs.columns)
    if missing_pairs:
        raise ValueError(f"pairs missing descriptor columns: {sorted(missing_pairs)}")
    required_features = {
        "name_token_jaccard",
        "name_char3_jaccard",
        "model_code_conflict",
        "number_conflict",
        "quantity_conflict",
        "attr_missing_any",
        "hard_negative_score",
    }
    missing_features = required_features - set(features.columns)
    if missing_features:
        raise ValueError(f"features missing descriptor columns: {sorted(missing_features)}")
    if len(pairs) != len(features):
        raise ValueError("pairs and features must have equal lengths")

    return pd.DataFrame(
        {
            "category": pairs["category"].astype(str).to_numpy(),
            "target": pd.to_numeric(pairs["target"], errors="raise").astype(int).to_numpy(),
            "lexical_bin": _bucket(features["name_token_jaccard"], [-1.1, 0.15, 0.35, 0.60, 0.80, 1.01]),
            "char_bin": _bucket(features["name_char3_jaccard"], [-1.1, 0.15, 0.35, 0.60, 0.80, 1.01]),
            "model_conflict": (features["model_code_conflict"].to_numpy(float) > 0).astype(int),
            "number_conflict": (features["number_conflict"].to_numpy(float) > 0).astype(int),
            "quantity_conflict": (features["quantity_conflict"].to_numpy(float) > 0).astype(int),
            "attr_missing": (features["attr_missing_any"].to_numpy(float) > 0).astype(int),
            "hard_negative_bin": _bucket(features["hard_negative_score"], [-1.1, 0.05, 0.20, 0.40, 0.70, 1.01]),
        }
    )


def development_fold_indices(manifest: dict, fold_id: int) -> tuple[np.ndarray, np.ndarray]:
    folds = manifest.get("fold_rows", [])
    if fold_id < 0 or fold_id >= len(folds):
        raise IndexError("fold_id out of range")
    valid = np.asarray(folds[fold_id], dtype=np.int64)
    train = np.asarray(
        sorted(row for idx, rows in enumerate(folds) if idx != fold_id for row in rows),
        dtype=np.int64,
    )
    gold = set(int(row) for row in manifest.get("gold_rows", []))
    if gold & set(train.tolist()) or gold & set(valid.tolist()):
        raise RuntimeError("gold rows leaked into development fold")
    return train, valid


def run_audit(
    items_path: Path,
    matches_path: Path,
    output_dir: Path,
    *,
    gold_fraction: float = 0.22,
    n_folds: int = 5,
    seed: int = 2026,
    chunk_size: int = 25_000,
    max_iter: int = 250,
) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    pairs = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    pairs = attach_pair_category(pairs, items).reset_index(drop=True)

    feature_started = time.perf_counter()
    features = build_features_v2_chunked(
        items,
        pairs,
        attribute_importance=None,
        chunk_size=chunk_size,
    )
    feature_seconds = time.perf_counter() - feature_started
    descriptors = build_split_descriptors(pairs, features)
    manifest = build_v5_split_manifest(
        pairs,
        descriptors,
        gold_fraction=gold_fraction,
        n_folds=n_folds,
        seed=seed,
    )
    split_sha = manifest_sha256(manifest)
    overlap_report = validate_manifest_no_overlap(pairs, manifest)

    fold_reports: list[dict] = []
    oof_rows: list[np.ndarray] = []
    oof_scores: list[np.ndarray] = []
    for fold_id in range(n_folds):
        train_idx, valid_idx = development_fold_indices(manifest, fold_id)
        x_train = features.iloc[train_idx].reset_index(drop=True)
        x_valid = features.iloc[valid_idx].reset_index(drop=True)
        train_pairs = pairs.iloc[train_idx].reset_index(drop=True)
        valid_pairs = pairs.iloc[valid_idx].reset_index(drop=True)
        weights = category_equalizing_weights(train_pairs["category"])

        fit_started = time.perf_counter()
        model = fit_estimator(
            x_train,
            train_pairs["target"].to_numpy(),
            sample_weight=weights,
            seed=seed + fold_id,
            max_iter=max_iter,
        )
        fit_seconds = time.perf_counter() - fit_started
        score = model.predict_proba(x_valid)[:, 1]
        report = macro_ap_report(valid_pairs, score)
        report.update(
            {
                "fold": fold_id,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "fit_seconds": float(fit_seconds),
            }
        )
        fold_reports.append(report)
        oof_rows.append(valid_idx)
        oof_scores.append(np.asarray(score, dtype=np.float64))

    row_index = np.concatenate(oof_rows)
    score_values = np.concatenate(oof_scores)
    order = np.argsort(row_index, kind="stable")
    row_index = row_index[order]
    score_values = score_values[order]
    dev_frame = pairs.iloc[row_index].reset_index(drop=True)
    oof_report = macro_ap_report(dev_frame, score_values)

    fold_values = np.asarray([r["macro_average_precision"] for r in fold_reports], dtype=float)
    payload = {
        "version": "v5a-validation-audit",
        "candidate": "human-structured-no-label-importance-oof",
        "split_sha256": split_sha,
        "rows_total": int(len(pairs)),
        "components_total": int(manifest["component_count"]),
        "gold_rows_sealed": int(len(manifest["gold_rows"])),
        "development_rows": int(sum(len(rows) for rows in manifest["fold_rows"])),
        "n_folds": int(n_folds),
        "validation_item_overlap": int(overlap_report["cross_split_item_overlap"]),
        "row_coverage": int(overlap_report["row_coverage"]),
        "feature_seconds": float(feature_seconds),
        "fold_macro_ap": fold_values.tolist(),
        "fold_mean_macro_ap": float(fold_values.mean()),
        "fold_median_macro_ap": float(np.median(fold_values)),
        "fold_std_macro_ap": float(fold_values.std()),
        "fold_worst_macro_ap": float(fold_values.min()),
        "development_oof_macro_ap": float(oof_report["macro_average_precision"]),
        "development_oof_per_category_ap": oof_report["per_category_ap"],
        "folds": fold_reports,
        "gold_metric_opened": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }

    (output_dir / "v5a-audit-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "split-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame({"row_index": row_index, "score": score_values}).to_parquet(
        output_dir / "development-oof.parquet", index=False
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-fraction", type=float, default=0.22)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--max-iter", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_audit(
        args.items,
        args.matches,
        args.output_dir,
        gold_fraction=args.gold_fraction,
        n_folds=args.n_folds,
        seed=args.seed,
        chunk_size=args.chunk_size,
        max_iter=args.max_iter,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
