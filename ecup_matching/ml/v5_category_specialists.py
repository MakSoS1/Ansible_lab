from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def fit_predict_category_specialists(
    train_features: pd.DataFrame,
    train_target,
    valid_features: pd.DataFrame,
    *,
    sample_weight=None,
    seed: int = 2026,
    max_iter: int = 300,
    min_samples_leaf: int = 15,
    l2_regularization: float = 2.0,
) -> np.ndarray:
    """Fit one compact HGB classifier per category and preserve validation order."""
    if "category" not in train_features.columns or "category" not in valid_features.columns:
        raise ValueError("train and validation features must contain category")
    y = np.asarray(train_target, dtype=np.int8)
    if len(y) != len(train_features):
        raise ValueError("train_target must match train_features")
    if sample_weight is None:
        weights = None
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if len(weights) != len(train_features):
            raise ValueError("sample_weight must match train_features")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sample_weight must be finite and non-negative")
    if max_iter <= 0 or min_samples_leaf <= 0 or l2_regularization <= 0:
        raise ValueError("model hyperparameters must be positive")
    numeric = [c for c in train_features.columns if c != "category"]
    if set(numeric) != {c for c in valid_features.columns if c != "category"}:
        raise ValueError("train/validation feature columns differ")

    train_category = train_features["category"].astype(str).to_numpy()
    valid_category = valid_features["category"].astype(str).to_numpy()
    scores = np.full(len(valid_features), np.nan, dtype=np.float64)

    for category in sorted(np.unique(valid_category).tolist()):
        train_mask = train_category == category
        valid_mask = valid_category == category
        if not train_mask.any():
            raise ValueError(f"validation category {category!r} has no training rows")
        category_y = y[train_mask]
        if len(np.unique(category_y)) < 2:
            raise ValueError(f"training category {category!r} does not contain both target classes")

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.07,
            max_iter=int(max_iter),
            max_leaf_nodes=31,
            min_samples_leaf=int(min_samples_leaf),
            l2_regularization=float(l2_regularization),
            early_stopping=False,
            random_state=int(seed),
        )
        fit_kwargs = {}
        if weights is not None:
            fit_kwargs["sample_weight"] = weights[train_mask]
        model.fit(
            train_features.loc[train_mask, numeric].to_numpy(dtype=np.float32),
            category_y,
            **fit_kwargs,
        )
        scores[valid_mask] = model.predict_proba(
            valid_features.loc[valid_mask, numeric].to_numpy(dtype=np.float32)
        )[:, 1]

    if not np.isfinite(scores).all():
        raise RuntimeError("category specialists did not score every validation row")
    return scores
