from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .v5_evaluation import macro_ap_report


def aggregate_oof_scores(
    frame: pd.DataFrame,
    base_scores,
    candidate_scores,
    fold_ids,
) -> dict[str, Any]:
    base = np.asarray(base_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    folds = np.asarray(fold_ids)
    if not (len(frame) == len(base) == len(candidate) == len(folds)):
        raise ValueError("frame, scores and fold_ids must have equal length")
    if not np.isfinite(base).all() or not np.isfinite(candidate).all():
        raise ValueError("OOF scores contain NaN or infinity")
    unique_folds = sorted(np.unique(folds).tolist())
    if len(unique_folds) < 2:
        raise ValueError("at least two folds are required")

    base_report = macro_ap_report(frame.reset_index(drop=True), base)
    candidate_report = macro_ap_report(frame.reset_index(drop=True), candidate)
    fold_reports: list[dict[str, Any]] = []
    for fold in unique_folds:
        mask = folds == fold
        fold_frame = frame.loc[mask].reset_index(drop=True)
        fold_base = macro_ap_report(fold_frame, base[mask])
        fold_candidate = macro_ap_report(fold_frame, candidate[mask])
        fold_reports.append(
            {
                "fold": int(fold) if isinstance(fold, (int, np.integer)) else str(fold),
                "valid_rows": int(mask.sum()),
                "base_macro_average_precision": float(fold_base["macro_average_precision"]),
                "macro_average_precision": float(fold_candidate["macro_average_precision"]),
                "delta_vs_base": float(
                    fold_candidate["macro_average_precision"] - fold_base["macro_average_precision"]
                ),
            }
        )

    return {
        "base_macro_average_precision": float(base_report["macro_average_precision"]),
        "macro_average_precision": float(candidate_report["macro_average_precision"]),
        "delta_vs_base": float(
            candidate_report["macro_average_precision"] - base_report["macro_average_precision"]
        ),
        "per_category_ap": candidate_report["per_category_ap"],
        "fold_reports": fold_reports,
    }
