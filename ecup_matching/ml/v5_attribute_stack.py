from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .v5_attribute_evidence import build_attribute_evidence_features, fit_attribute_evidence
from .v5_evaluation import macro_ap_report
from .v5_residual import clipped_logit


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    z = np.exp(x[~pos])
    out[~pos] = z / (1.0 + z)
    return out


def crossfit_attribute_evidence_stack(
    items: pd.DataFrame,
    frame: pd.DataFrame,
    base_scores,
    fold_ids,
    *,
    min_support: int = 20,
    smoothing: float = 2.0,
    evidence_clip: float = 8.0,
    seed: int = 2026,
) -> dict[str, Any]:
    """Apply train-fold-only attribute likelihood evidence to held folds.

    `attr_evidence_sum` is a sum of per-key log likelihood ratios, so the
    statistically natural fixed combination is additive in log-odds space.
    No meta-model, category alpha, or held-fold calibration is fitted here.
    """
    del seed  # retained in the public interface for experiment-manifest symmetry
    required = {"id1", "id2", "target", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    base = np.asarray(base_scores, dtype=np.float64)
    folds = np.asarray(fold_ids)
    if not (len(frame) == len(base) == len(folds)):
        raise ValueError("frame, base_scores and fold_ids must have equal length")
    unique_folds = sorted(np.unique(folds).tolist())
    if len(unique_folds) < 2:
        raise ValueError("at least two folds are required")
    if evidence_clip <= 0:
        raise ValueError("evidence_clip must be positive")

    scores = np.full(len(frame), np.nan, dtype=np.float64)
    evidence_scores = np.full(len(frame), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []

    for fold_value in unique_folds:
        valid_mask = folds == fold_value
        train_mask = ~valid_mask
        train = frame.loc[train_mask, ["id1", "id2", "target"]].reset_index(drop=True)
        valid = frame.loc[valid_mask, ["id1", "id2"]].reset_index(drop=True)
        learned = fit_attribute_evidence(
            items,
            train,
            min_support=min_support,
            smoothing=smoothing,
        )
        held_features = build_attribute_evidence_features(items, valid, learned)
        held_evidence = held_features["attr_evidence_sum"].to_numpy(dtype=np.float64)
        held_evidence = np.clip(held_evidence, -float(evidence_clip), float(evidence_clip))
        held_score = _expit(clipped_logit(base[valid_mask]) + held_evidence)
        scores[valid_mask] = held_score
        evidence_scores[valid_mask] = held_evidence

        valid_frame = frame.loc[valid_mask].reset_index(drop=True)
        base_report = macro_ap_report(valid_frame, base[valid_mask])
        candidate_report = macro_ap_report(valid_frame, held_score)
        fold_reports.append(
            {
                "fold": int(fold_value) if isinstance(fold_value, (int, np.integer)) else str(fold_value),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "learned_categories": int(len(learned)),
                "learned_keys": int(sum(len(keys) for keys in learned.values())),
                "base_macro_average_precision": float(base_report["macro_average_precision"]),
                "macro_average_precision": float(candidate_report["macro_average_precision"]),
                "delta_vs_base": float(candidate_report["macro_average_precision"] - base_report["macro_average_precision"]),
            }
        )

    if not np.isfinite(scores).all() or not np.isfinite(evidence_scores).all():
        raise RuntimeError("attribute stack failed to score every development row")
    base_report = macro_ap_report(frame.reset_index(drop=True), base)
    candidate_report = macro_ap_report(frame.reset_index(drop=True), scores)
    return {
        "scores": scores,
        "evidence": evidence_scores,
        "base_macro_average_precision": float(base_report["macro_average_precision"]),
        "macro_average_precision": float(candidate_report["macro_average_precision"]),
        "delta_vs_base": float(candidate_report["macro_average_precision"] - base_report["macro_average_precision"]),
        "per_category_ap": candidate_report["per_category_ap"],
        "fold_reports": fold_reports,
    }
