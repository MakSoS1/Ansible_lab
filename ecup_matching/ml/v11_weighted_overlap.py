from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


@dataclass(frozen=True)
class OverlapConfig:
    n_features: int = 1 << 16
    ngram_range: tuple[int, int] = (3, 5)
    local_blend: float = 0.40
    min_category_rows: int = 1200


def _texts(items: pd.DataFrame) -> list[str]:
    names = items["name"].fillna("").astype(str).tolist()
    attrs = items["attributes"].map(lambda x: "" if x is None else str(x)).tolist()
    return [f"{name} | {attr}" for name, attr in zip(names, attrs, strict=True)]


def _vectorizer(config: OverlapConfig) -> HashingVectorizer:
    return HashingVectorizer(n_features=int(config.n_features), alternate_sign=False, norm="l2", lowercase=True, analyzer="char_wb", ngram_range=config.ngram_range, dtype=np.float32)


def build_item_matrix(items: pd.DataFrame, config: OverlapConfig) -> tuple[sparse.csr_matrix, dict[object, int]]:
    required = {"id", "name", "attributes"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items missing required columns: {sorted(missing)}")
    ids = items["id"].tolist()
    index = {item_id: row for row, item_id in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("items.id must be unique")
    return _vectorizer(config).transform(_texts(items)).tocsr(), index


def build_pair_interactions(matrix: sparse.csr_matrix, index: dict[object, int], pairs: pd.DataFrame) -> sparse.csr_matrix:
    if not {"id1", "id2"}.issubset(pairs.columns):
        raise ValueError("pairs must contain id1 and id2")
    try:
        left = np.fromiter((index[x] for x in pairs["id1"]), dtype=np.int64, count=len(pairs))
        right = np.fromiter((index[x] for x in pairs["id2"]), dtype=np.int64, count=len(pairs))
    except KeyError as exc:
        raise KeyError(f"pair references missing item: {exc.args[0]!r}") from exc
    return matrix[left].multiply(matrix[right]).tocsr()


def _fit_linear(x: sparse.csr_matrix, y: np.ndarray) -> SGDClassifier:
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=2e-5, max_iter=60, tol=1e-4, class_weight="balanced", random_state=20260813, average=True)
    model.fit(x, y)
    return model


def fit_models(interactions: sparse.csr_matrix, target: np.ndarray, categories: np.ndarray, *, min_category_rows: int, local_blend: float) -> dict:
    target = np.asarray(target, dtype=np.int8)
    categories = np.asarray(categories, dtype=str)
    if interactions.shape[0] != len(target) or len(target) != len(categories):
        raise ValueError("interaction/target/category length mismatch")
    global_model = _fit_linear(interactions, target)
    local_models = {}
    for category in sorted(np.unique(categories).tolist()):
        mask = categories == category
        if int(mask.sum()) >= int(min_category_rows) and len(np.unique(target[mask])) >= 2:
            local_models[category] = _fit_linear(interactions[mask], target[mask])
    return {"global_model": global_model, "local_models": local_models, "local_blend": float(local_blend)}


def predict_models(bundle: dict, interactions: sparse.csr_matrix, categories: np.ndarray) -> np.ndarray:
    categories = np.asarray(categories, dtype=str)
    global_score = bundle["global_model"].predict_proba(interactions)[:, 1].astype(np.float64)
    result = global_score.copy()
    blend = float(bundle.get("local_blend", 0.40))
    for category, model in bundle.get("local_models", {}).items():
        mask = categories == category
        if mask.any():
            local = model.predict_proba(interactions[mask])[:, 1]
            result[mask] = (1.0 - blend) * global_score[mask] + blend * local
    return result


def fit_weighted_overlap(items: pd.DataFrame, train_pairs: pd.DataFrame, *, n_features: int = 1 << 16, min_category_rows: int = 1200, local_blend: float = 0.40) -> dict:
    required = {"id1", "id2", "target", "category"}
    missing = required - set(train_pairs.columns)
    if missing:
        raise ValueError(f"train_pairs missing required columns: {sorted(missing)}")
    config = OverlapConfig(n_features=n_features, min_category_rows=min_category_rows, local_blend=local_blend)
    matrix, index = build_item_matrix(items, config)
    interactions = build_pair_interactions(matrix, index, train_pairs)
    models = fit_models(interactions, train_pairs["target"].to_numpy(np.int8), train_pairs["category"].astype(str).to_numpy(), min_category_rows=min_category_rows, local_blend=local_blend)
    return {"config": config, **models}


def predict_weighted_overlap(bundle: dict, items: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    if "category" not in pairs.columns:
        raise ValueError("pairs must contain category")
    matrix, index = build_item_matrix(items, bundle["config"])
    interactions = build_pair_interactions(matrix, index, pairs)
    return predict_models(bundle, interactions, pairs["category"].astype(str).to_numpy())


def crossfit_weighted_overlap(items: pd.DataFrame, pairs: pd.DataFrame, folds: np.ndarray, *, n_features: int = 1 << 16, min_category_rows: int = 1200, local_blend: float = 0.40) -> np.ndarray:
    config = OverlapConfig(n_features=n_features, min_category_rows=min_category_rows, local_blend=local_blend)
    matrix, index = build_item_matrix(items, config)
    interactions = build_pair_interactions(matrix, index, pairs)
    y = pairs["target"].to_numpy(np.int8)
    categories = pairs["category"].astype(str).to_numpy()
    folds = np.asarray(folds)
    result = np.full(len(pairs), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds).tolist()):
        valid = folds == fold
        train = ~valid
        models = fit_models(interactions[train], y[train], categories[train], min_category_rows=min_category_rows, local_blend=local_blend)
        result[valid] = predict_models(models, interactions[valid], categories[valid])
    if not np.isfinite(result).all():
        raise RuntimeError("weighted-overlap crossfit produced non-finite scores")
    return result


__all__ = ["OverlapConfig", "build_item_matrix", "build_pair_interactions", "fit_models", "predict_models", "fit_weighted_overlap", "predict_weighted_overlap", "crossfit_weighted_overlap"]
