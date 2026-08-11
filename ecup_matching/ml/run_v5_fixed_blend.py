from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_evaluation import macro_ap_report
from .v5_fixed_blend import fixed_blend_candidates
from .v5_validation import manifest_sha256


def align_oof_frame(
    paths: Iterable[Path],
    *,
    expected_rows,
    expected_folds=None,
    required_columns: tuple[str, ...] = ("score",),
    source_name: str,
) -> pd.DataFrame:
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError(f"{source_name} has no OOF files")
    pieces = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(pieces, ignore_index=True)

    required = {"row_index", *required_columns}
    if expected_folds is not None:
        required.add("fold")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")
    if frame["row_index"].duplicated().any():
        raise ValueError(f"{source_name} contains duplicate OOF row_index values")

    frame = frame.sort_values("row_index").reset_index(drop=True)
    rows = np.asarray(expected_rows, dtype=np.int64)
    actual_rows = frame["row_index"].astype(np.int64).to_numpy()
    if not np.array_equal(actual_rows, rows):
        raise ValueError(f"{source_name} OOF rows do not exactly cover sealed development rows")

    for column in required_columns:
        numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{source_name} column {column!r} must contain only finite values")

    if expected_folds is not None:
        folds = np.asarray(expected_folds, dtype=np.int16)
        actual_folds = frame["fold"].astype(np.int16).to_numpy()
        if not np.array_equal(actual_folds, folds):
            raise ValueError(f"{source_name} OOF fold IDs do not match sealed manifest")
    return frame


def _fold_reports(
    frame: pd.DataFrame,
    folds: np.ndarray,
    explicit_scores: np.ndarray,
    candidate_scores: np.ndarray,
) -> list[dict]:
    reports: list[dict] = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        fold_frame = frame.loc[mask].reset_index(drop=True)
        explicit = macro_ap_report(fold_frame, explicit_scores[mask])["macro_average_precision"]
        candidate = macro_ap_report(fold_frame, candidate_scores[mask])["macro_average_precision"]
        reports.append(
            {
                "fold": int(fold),
                "rows": int(mask.sum()),
                "explicit_macro_average_precision": float(explicit),
                "macro_average_precision": float(candidate),
                "delta_vs_explicit": float(candidate - explicit),
            }
        )
    return reports


def run_fixed_blend(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    category_oof_path: Path,
    weak_oof_path: Path,
    sparse_fold_dir: Path,
    explicit_fold_dir: Path,
    contrastive_oof_path: Path,
    output_dir: Path,
    expected_split_sha: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_split_sha = manifest_sha256(manifest)
    if actual_split_sha != expected_split_sha:
        raise ValueError(f"sealed split SHA mismatch: {actual_split_sha}")

    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    dev_rows, folds = development_rows_and_folds(manifest, total_rows=len(matches))
    folds = np.asarray(folds, dtype=np.int16)

    category = align_oof_frame(
        [category_oof_path],
        expected_rows=dev_rows,
        required_columns=("score",),
        source_name="category",
    )
    weak = align_oof_frame(
        [weak_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("score",),
        source_name="weak",
    )
    sparse = align_oof_frame(
        sorted(Path(sparse_fold_dir).rglob("v5-sparse-fold-*-oof.parquet")),
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("score",),
        source_name="sparse",
    )
    explicit = align_oof_frame(
        sorted(Path(explicit_fold_dir).rglob("v5-explicit-fold-*-oof.parquet")),
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("score",),
        source_name="explicit",
    )
    contrastive = align_oof_frame(
        [contrastive_oof_path],
        expected_rows=dev_rows,
        expected_folds=folds,
        required_columns=("embedding_cosine",),
        source_name="contrastive",
    )

    dev = matches.iloc[dev_rows].reset_index(drop=True)
    wanted_ids = pd.unique(pd.concat([dev["id1"], dev["id2"]], ignore_index=True))
    items = select_items_by_ids(items_path, wanted_ids, include_attributes=False)
    category_by_id = items.set_index("id")["category"].astype(str)
    dev["category"] = dev["id1"].map(category_by_id)
    if dev["category"].isna().any():
        raise RuntimeError("failed to attach official categories to development rows")

    source_scores = {
        "category": category["score"].to_numpy(dtype=np.float64),
        "weak": weak["score"].to_numpy(dtype=np.float64),
        "sparse": sparse["score"].to_numpy(dtype=np.float64),
        "explicit": explicit["score"].to_numpy(dtype=np.float64),
    }
    cosine = contrastive["embedding_cosine"].to_numpy(dtype=np.float64)
    candidates = fixed_blend_candidates(source_scores, contrastive_cosine=cosine)

    source_metrics = {
        name: float(macro_ap_report(dev, score, strict_official=True)["macro_average_precision"])
        for name, score in source_scores.items()
    }
    source_metrics["contrastive_cosine"] = float(
        macro_ap_report(dev, cosine, strict_official=True)["macro_average_precision"]
    )
    explicit_ap = source_metrics["explicit"]

    candidate_reports: dict[str, dict] = {}
    eligible: list[str] = []
    for name, score in candidates.items():
        report = macro_ap_report(dev, score, strict_official=True)
        folds_report = _fold_reports(dev, folds, source_scores["explicit"], score)
        min_fold_delta = min(row["delta_vs_explicit"] for row in folds_report)
        candidate_ap = float(report["macro_average_precision"])
        keep_eligible = bool(candidate_ap > explicit_ap and min_fold_delta >= -0.001)
        if keep_eligible:
            eligible.append(name)
        candidate_reports[name] = {
            "macro_average_precision": candidate_ap,
            "delta_vs_explicit": float(candidate_ap - explicit_ap),
            "min_fold_delta_vs_explicit": float(min_fold_delta),
            "keep_eligible": keep_eligible,
            "fold_reports": folds_report,
            "per_category_ap": report["per_category_ap"],
        }

    best_name = max(candidate_reports, key=lambda name: candidate_reports[name]["macro_average_precision"])
    best_report = candidate_reports[best_name]
    payload = {
        "version": "v5-fixed-label-free-blend",
        "split_sha256": expected_split_sha,
        "development_rows": int(len(dev)),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "target_fitted_blender": False,
        "predeclared_candidate_count": int(len(candidate_reports)),
        "explicit_anchor_macro_average_precision": float(explicit_ap),
        "source_metrics": source_metrics,
        "candidates": candidate_reports,
        "best_observed_name": best_name,
        "best_observed_macro_average_precision": float(best_report["macro_average_precision"]),
        "best_observed_delta_vs_explicit": float(best_report["delta_vs_explicit"]),
        "keep_eligible_candidates": eligible,
    }
    (output_dir / "v5-fixed-blend-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    output = pd.DataFrame(
        {
            "row_index": dev_rows,
            "fold": folds,
            **{f"source_{name}": values for name, values in source_scores.items()},
            "source_contrastive_cosine": cosine,
            **{f"candidate_{name}": values for name, values in candidates.items()},
        }
    )
    output.to_parquet(output_dir / "v5-fixed-blend-oof.parquet", index=False)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--category-oof", type=Path, required=True)
    parser.add_argument("--weak-oof", type=Path, required=True)
    parser.add_argument("--sparse-fold-dir", type=Path, required=True)
    parser.add_argument("--explicit-fold-dir", type=Path, required=True)
    parser.add_argument("--contrastive-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    args = parser.parse_args()
    payload = run_fixed_blend(
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        category_oof_path=args.category_oof,
        weak_oof_path=args.weak_oof,
        sparse_fold_dir=args.sparse_fold_dir,
        explicit_fold_dir=args.explicit_fold_dir,
        contrastive_oof_path=args.contrastive_oof,
        output_dir=args.output_dir,
        expected_split_sha=args.expected_split_sha,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
