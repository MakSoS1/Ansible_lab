from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .train_v1 import category_equalizing_weights
from .v5_evaluation import macro_ap_report


def _stack_features(
    frame: pd.DataFrame,
    base_scores: np.ndarray,
    semantic_features: pd.DataFrame,
) -> pd.DataFrame:
    if not {"target", "category"}.issubset(frame.columns):
        raise ValueError("frame must contain target and category")
    if len(frame) != len(base_scores) or len(frame) != len(semantic_features):
        raise ValueError("frame, base_scores and semantic_features must have equal length")
    semantic = semantic_features.reset_index(drop=True).copy()
    for column in semantic.columns:
        semantic[column] = pd.to_numeric(semantic[column], errors="raise").astype(np.float32)
    if not np.isfinite(semantic.to_numpy(dtype=np.float64)).all():
        raise ValueError("semantic_features contain NaN or infinity")

    base = np.asarray(base_scores, dtype=np.float64)
    if not np.isfinite(base).all():
        raise ValueError("base_scores contain NaN or infinity")
    base = np.clip(base, 1e-6, 1.0 - 1e-6)
    out = semantic
    out.insert(0, "base_score", base.astype(np.float32))
    out.insert(1, "base_logit", (np.log(base) - np.log1p(-base)).astype(np.float32))

    categories = pd.get_dummies(
        frame["category"].astype(str).reset_index(drop=True),
        prefix="category",
        dtype=np.float32,
    )
    return pd.concat([out, categories], axis=1)


def crossfit_semantic_stack(
    frame: pd.DataFrame,
    base_scores,
    semantic_features: pd.DataFrame,
    fold_ids,
    *,
    seed: int = 2026,
    max_iter: int = 180,
) -> dict[str, Any]:
    """Train a second-level semantic stack strictly out of fold.

    The base scores supplied to this function must themselves be OOF. For every
    held fold, the second-level estimator is fit only on rows from other folds.
    """
    base = np.asarray(base_scores, dtype=np.float64)
    folds = np.asarray(fold_ids)
    if len(frame) != len(base) or len(frame) != len(semantic_features) or len(frame) != len(folds):
        raise ValueError("frame, scores, features and fold_ids must have equal length")
    unique_folds = sorted(np.unique(folds).tolist())
    if len(unique_folds) < 2:
        raise ValueError("at least two folds are required")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    x = _stack_features(frame.reset_index(drop=True), base, semantic_features)
    y = pd.to_numeric(frame["target"], errors="raise").astype(np.int8).to_numpy()
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []

    for fold_number, fold_value in enumerate(unique_folds):
        valid_mask = folds == fold_value
        train_mask = ~valid_mask
        if not valid_mask.any() or not train_mask.any():
            raise ValueError("every fold must have both train and validation rows")
        if len(np.unique(y[train_mask])) < 2:
            raise ValueError("training partition must contain both target classes")

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=int(max_iter),
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=8.0,
            random_state=seed + fold_number,
        )
        weights = category_equalizing_weights(frame.loc[train_mask, "category"])
        model.fit(x.loc[train_mask], y[train_mask], sample_weight=weights)
        fold_score = model.predict_proba(x.loc[valid_mask])[:, 1]
        scores[valid_mask] = fold_score

        valid_frame = frame.loc[valid_mask].reset_index(drop=True)
        candidate_report = macro_ap_report(valid_frame, fold_score)
        base_report = macro_ap_report(valid_frame, base[valid_mask])
        fold_reports.append(
            {
                "fold": int(fold_value) if isinstance(fold_value, (int, np.integer)) else str(fold_value),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "base_macro_average_precision": float(base_report["macro_average_precision"]),
                "macro_average_precision": float(candidate_report["macro_average_precision"]),
                "delta_vs_base": float(
                    candidate_report["macro_average_precision"] - base_report["macro_average_precision"]
                ),
            }
        )

    if not np.isfinite(scores).all():
        raise RuntimeError("semantic stack did not produce a score for every row")
    candidate_report = macro_ap_report(frame.reset_index(drop=True), scores)
    base_report = macro_ap_report(frame.reset_index(drop=True), base)
    return {
        "scores": scores,
        "base_macro_average_precision": float(base_report["macro_average_precision"]),
        "macro_average_precision": float(candidate_report["macro_average_precision"]),
        "delta_vs_base": float(
            candidate_report["macro_average_precision"] - base_report["macro_average_precision"]
        ),
        "per_category_ap": candidate_report["per_category_ap"],
        "fold_reports": fold_reports,
    }
