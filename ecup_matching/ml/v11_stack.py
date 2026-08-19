from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


GLOBAL_PARAMS = dict(
    learning_rate=0.055,
    max_iter=220,
    max_leaf_nodes=31,
    max_depth=6,
    min_samples_leaf=80,
    l2_regularization=4.0,
    early_stopping=False,
    random_state=20260813,
)
LOCAL_PARAMS = dict(
    learning_rate=0.06,
    max_iter=180,
    max_leaf_nodes=15,
    max_depth=5,
    min_samples_leaf=40,
    l2_regularization=5.0,
    early_stopping=False,
    random_state=20260813,
)


def _numeric_names(features: pd.DataFrame) -> list[str]:
    names = [c for c in features.columns if c != "category"]
    if not names:
        raise ValueError("at least one numeric feature is required")
    return names


def _category_weights(categories: np.ndarray) -> np.ndarray:
    cats = np.asarray(categories, dtype=str)
    unique, counts = np.unique(cats, return_counts=True)
    inv = {cat: len(cats) / (len(unique) * count) for cat, count in zip(unique, counts)}
    return np.asarray([inv[x] for x in cats], dtype=np.float64)


def fit_hgb_bundle(
    features: pd.DataFrame,
    target: np.ndarray,
    *,
    min_local_rows: int = 1200,
    local_blend: float = 0.35,
) -> dict:
    target = np.asarray(target, dtype=np.int8)
    if len(features) != len(target):
        raise ValueError("feature/target length mismatch")
    names = _numeric_names(features)
    cats = features["category"].astype(str).to_numpy()
    category_values = sorted(np.unique(cats).tolist())
    cat_to_code = {cat: i for i, cat in enumerate(category_values)}
    cat_code = np.asarray([cat_to_code[x] for x in cats], dtype=np.float64)
    numeric = features[names].to_numpy(dtype=np.float32, copy=False)
    matrix = np.column_stack([numeric, cat_code])
    global_model = HistGradientBoostingClassifier(
        **GLOBAL_PARAMS,
        categorical_features=[len(names)],
        class_weight="balanced",
    )
    global_model.fit(matrix, target, sample_weight=_category_weights(cats))

    local_models: dict[str, HistGradientBoostingClassifier] = {}
    for cat in category_values:
        mask = cats == cat
        y = target[mask]
        if int(mask.sum()) < int(min_local_rows) or len(np.unique(y)) < 2:
            continue
        model = HistGradientBoostingClassifier(**LOCAL_PARAMS, class_weight="balanced")
        model.fit(numeric[mask], y)
        local_models[cat] = model
    return {
        "numeric_features": names,
        "category_values": category_values,
        "global_model": global_model,
        "local_models": local_models,
        "local_blend": float(local_blend),
        "min_local_rows": int(min_local_rows),
    }


def predict_hgb_bundle(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    names = list(bundle["numeric_features"])
    cats = features["category"].astype(str).to_numpy()
    mapping = {cat: i for i, cat in enumerate(bundle["category_values"])}
    cat_code = np.asarray([mapping.get(x, np.nan) for x in cats], dtype=np.float64)
    numeric = features[names].to_numpy(dtype=np.float32, copy=False)
    matrix = np.column_stack([numeric, cat_code])
    global_score = bundle["global_model"].predict_proba(matrix)[:, 1].astype(np.float64)
    result = global_score.copy()
    blend = float(bundle.get("local_blend", 0.35))
    for cat, model in bundle.get("local_models", {}).items():
        mask = cats == cat
        if not mask.any():
            continue
        local = model.predict_proba(numeric[mask])[:, 1]
        result[mask] = (1.0 - blend) * global_score[mask] + blend * local
    return result


def crossfit_hgb_scores(
    features: pd.DataFrame,
    target: np.ndarray,
    folds: np.ndarray,
    *,
    min_local_rows: int = 1200,
    local_blend: float = 0.35,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.int8)
    folds = np.asarray(folds)
    if len(features) != len(target) or len(target) != len(folds):
        raise ValueError("crossfit input length mismatch")
    result = np.full(len(features), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds).tolist()):
        valid = folds == fold
        train = ~valid
        bundle = fit_hgb_bundle(
            features.loc[train].reset_index(drop=True),
            target[train],
            min_local_rows=min_local_rows,
            local_blend=local_blend,
        )
        result[valid] = predict_hgb_bundle(bundle, features.loc[valid].reset_index(drop=True))
    if not np.isfinite(result).all():
        raise RuntimeError("crossfit produced incomplete/non-finite scores")
    return result


__all__ = ["fit_hgb_bundle", "predict_hgb_bundle", "crossfit_hgb_scores"]
