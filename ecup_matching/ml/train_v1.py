from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .batch_features import build_features_chunked
from .features import FEATURE_NAMES
from .metrics import macro_average_precision
from .model_io import save_model_bundle
from .split import component_split


SEED = 2026


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:  # sklearn < 1.2 compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def train_estimator(
    features: pd.DataFrame,
    y,
    sample_weight=None,
    max_iter: int = 350,
) -> Pipeline:
    if "category" not in features.columns:
        raise ValueError("features must contain category")
    numeric = [c for c in features.columns if c != "category"]
    preprocess = ColumnTransformer(
        [("category", _one_hot_encoder(), ["category"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.07,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=SEED,
    )
    model = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
    model.fit(features[["category", *numeric]], np.asarray(y, dtype=int), **fit_kwargs)
    return model


def category_equalizing_weights(categories: pd.Series) -> np.ndarray:
    counts = categories.astype(str).value_counts()
    weights = categories.astype(str).map(lambda c: 1.0 / float(counts[c])).to_numpy(float)
    return weights / weights.mean()


def attach_pair_category(matches: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    category_map = items.set_index("id")["category"]
    result = matches.copy()
    result["category"] = result["id1"].map(category_map)
    category2 = result["id2"].map(category_map)
    if result["category"].isna().any() or category2.isna().any():
        raise ValueError("matches reference item IDs absent from items")
    cross = result["category"].astype(str) != category2.astype(str)
    if cross.any():
        raise ValueError(f"found {int(cross.sum())} cross-category training pairs")
    result["category"] = result["category"].astype(str)
    return result


def _item_overlap(matches: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray) -> int:
    train = matches.iloc[train_idx]
    valid = matches.iloc[valid_idx]
    train_items = set(train["id1"]) | set(train["id2"])
    valid_items = set(valid["id1"]) | set(valid["id2"])
    return len(train_items & valid_items)


def render_metrics_markdown(metrics: dict) -> str:
    lines = [
        "# E-CUP v1 validation",
        "",
        f"- Macro Average Precision: **{metrics['macro_average_precision']:.6f}**",
        f"- train rows: {metrics['train_rows']:,}",
        f"- valid rows: {metrics['valid_rows']:,}",
        f"- item overlap: {metrics['item_overlap']}",
        f"- feature seconds: {metrics['feature_seconds']:.1f}",
        f"- fit seconds: {metrics['fit_seconds']:.1f}",
        "",
        "## Per category AP",
        "",
        "| category | AP |",
        "|---|---:|",
    ]
    for cat, score in metrics["per_category_ap"].items():
        lines.append(f"| {cat} | {score:.6f} |")
    return "\n".join(lines) + "\n"


def train_full(
    items_path: Path,
    matches_path: Path,
    model_out: Path,
    manifest_out: Path,
    metrics_out: Path,
    metrics_md_out: Path,
    valid_pred_out: Path,
    valid_fraction: float = 0.2,
    chunk_size: int = 25_000,
    max_iter: int = 350,
) -> dict:
    started = time.perf_counter()
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(matches_path)
    if "target" not in matches.columns:
        raise ValueError("training matches must contain target")
    matches = attach_pair_category(matches, items)

    train_idx, valid_idx = component_split(matches, valid_fraction=valid_fraction, seed=SEED)
    overlap = _item_overlap(matches, train_idx, valid_idx)
    if overlap != 0:
        raise RuntimeError(f"item leakage detected: {overlap} IDs overlap")

    feature_started = time.perf_counter()
    train_pairs = matches.iloc[train_idx].reset_index(drop=True)
    valid_pairs = matches.iloc[valid_idx].reset_index(drop=True)
    x_train = build_features_chunked(items, train_pairs, chunk_size=chunk_size)
    x_valid = build_features_chunked(items, valid_pairs, chunk_size=chunk_size)
    feature_seconds = time.perf_counter() - feature_started

    weights = category_equalizing_weights(train_pairs["category"])
    fit_started = time.perf_counter()
    model = train_estimator(x_train, train_pairs["target"].to_numpy(), weights, max_iter=max_iter)
    fit_seconds = time.perf_counter() - fit_started

    valid_score = model.predict_proba(x_valid)[:, 1]
    if not np.isfinite(valid_score).all():
        raise RuntimeError("validation predictions contain NaN/Inf")
    macro_ap, per_cat = macro_average_precision(
        valid_pairs["target"].to_numpy(), valid_score, valid_pairs["category"].to_numpy()
    )

    metrics = {
        "version": "v1-structured-hgb",
        "seed": SEED,
        "macro_average_precision": macro_ap,
        "per_category_ap": per_cat,
        "train_rows": int(len(train_pairs)),
        "valid_rows": int(len(valid_pairs)),
        "valid_fraction_actual": float(len(valid_pairs) / len(matches)),
        "item_overlap": int(overlap),
        "feature_seconds": float(feature_seconds),
        "fit_seconds": float(fit_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "max_iter": int(max_iter),
        "feature_names": list(FEATURE_NAMES),
    }
    manifest = {
        "version": "v1-structured-hgb",
        "seed": SEED,
        "feature_names": list(FEATURE_NAMES),
        "model_class": "sklearn.ensemble.HistGradientBoostingClassifier",
        "validation_macro_ap": macro_ap,
    }
    save_model_bundle(model, model_out, manifest_out, manifest)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_md_out.write_text(render_metrics_markdown(metrics), encoding="utf-8")

    pred = valid_pairs[["id1", "id2", "target", "category"]].copy()
    pred["predict"] = valid_score
    valid_pred_out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(valid_pred_out, index=False)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--matches", required=True, type=Path)
    parser.add_argument("--model-out", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--metrics-out", required=True, type=Path)
    parser.add_argument("--metrics-md-out", required=True, type=Path)
    parser.add_argument("--valid-pred-out", required=True, type=Path)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--max-iter", type=int, default=350)
    args = parser.parse_args()

    metrics = train_full(
        args.items,
        args.matches,
        args.model_out,
        args.manifest_out,
        args.metrics_out,
        args.metrics_md_out,
        args.valid_pred_out,
        valid_fraction=args.valid_fraction,
        chunk_size=args.chunk_size,
        max_iter=args.max_iter,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
