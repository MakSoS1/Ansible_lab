from __future__ import annotations

import numpy as np
import pandas as pd


def _validate(frame: pd.DataFrame, scores) -> np.ndarray:
    required = {"id1", "id2", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"graph frame missing columns: {sorted(missing)}")
    score = np.asarray(scores, dtype=np.float64)
    if score.ndim != 1 or len(score) != len(frame):
        raise ValueError("scores must be one-dimensional and aligned with frame")
    if len(score) == 0:
        raise ValueError("graph scorer requires at least one edge")
    if not np.isfinite(score).all():
        raise ValueError("scores contain NaN or infinity")
    return score


def _undirected_endpoint_table(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    rows = np.arange(len(frame), dtype=np.int64)
    categories = frame["category"].astype(str).to_numpy()
    left = pd.DataFrame(
        {
            "_row": rows,
            "_side": "left",
            "category": categories,
            "item": frame["id1"].to_numpy(),
            "score": score,
        }
    )
    right = pd.DataFrame(
        {
            "_row": rows,
            "_side": "right",
            "category": categories,
            "item": frame["id2"].to_numpy(),
            "score": score,
        }
    )
    endpoint = pd.concat([left, right], ignore_index=True)
    group = endpoint.groupby(["category", "item"], sort=False, dropna=False)
    endpoint["degree"] = group["score"].transform("size").astype(np.int32)
    # Exact score ties are equally-best. No row-order or label based tie break.
    endpoint["rank"] = group["score"].rank(method="min", ascending=False).astype(np.int32)
    endpoint["best_score"] = group["score"].transform("max").astype(float)
    endpoint["gap_to_best"] = (endpoint["best_score"] - endpoint["score"]).clip(lower=0.0)
    degree = endpoint["degree"].to_numpy(dtype=np.float64)
    rank = endpoint["rank"].to_numpy(dtype=np.float64)
    denom = np.maximum(1.0, degree - 1.0)
    endpoint["endpoint_percentile"] = np.clip(
        np.where(degree <= 1.0, 1.0, 1.0 - (rank - 1.0) / denom),
        0.0,
        1.0,
    )
    return endpoint


def _side_features(endpoint: pd.DataFrame, side: str) -> pd.DataFrame:
    part = endpoint.loc[endpoint["_side"] == side].sort_values("_row")
    if len(part) == 0:
        raise RuntimeError(f"missing {side} endpoints")
    prefix = side
    out = part.set_index("_row")[["degree", "rank", "best_score", "gap_to_best", "endpoint_percentile"]].copy()
    out.columns = [
        f"degree_{prefix}",
        f"rank_{prefix}",
        f"best_score_{prefix}",
        f"gap_to_best_{prefix}",
        f"endpoint_percentile_{prefix}",
    ]
    return out


def graph_features(frame: pd.DataFrame, scores) -> pd.DataFrame:
    """Compute target-free observed-edge context on an undirected category graph.

    An item endpoint sees all of its incident edges regardless of whether it
    appears in ``id1`` or ``id2``. Missing edges are never synthesized. The
    graph is scoped by category even if an identifier appears in multiple
    categories.
    """
    score = _validate(frame, scores)
    endpoint = _undirected_endpoint_table(frame, score)
    left = _side_features(endpoint, "left")
    right = _side_features(endpoint, "right")
    out = pd.concat([left, right], axis=1).sort_index().reset_index(drop=True)

    out["reciprocal_best"] = (
        (out["rank_left"].to_numpy() == 1)
        & (out["rank_right"].to_numpy() == 1)
    )
    out["reciprocal_top3"] = (
        (out["rank_left"].to_numpy() <= 3)
        & (out["rank_right"].to_numpy() <= 3)
    )
    out["min_endpoint_percentile"] = np.minimum(
        out["endpoint_percentile_left"].to_numpy(float),
        out["endpoint_percentile_right"].to_numpy(float),
    )
    out["max_endpoint_degree"] = np.maximum(
        out["degree_left"].to_numpy(np.int32),
        out["degree_right"].to_numpy(np.int32),
    )
    out["ambiguity_log_degree"] = np.log1p(
        np.maximum(0.0, out["max_endpoint_degree"].to_numpy(float) - 1.0)
    )

    # A category-local percentile is a monotone transform of the original score
    # before graph context is added, preserving base AP ordering by itself.
    category = frame["category"].astype(str).reset_index(drop=True)
    score_series = pd.Series(score)
    category_count = category.groupby(category, sort=False).transform("size").to_numpy(float)
    category_rank = score_series.groupby(category, sort=False).rank(
        method="average", ascending=False
    ).to_numpy(float)
    denom = np.maximum(1.0, category_count - 1.0)
    out["category_score_percentile"] = np.clip(
        np.where(
            category_count <= 1.0,
            1.0,
            1.0 - (category_rank - 1.0) / denom,
        ),
        0.0,
        1.0,
    )

    if not np.isfinite(out.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise RuntimeError("graph feature computation produced nonfinite values")
    return out


def graph_rescore(
    scores,
    features: pd.DataFrame,
    *,
    reciprocal_best_bonus: float = 0.03,
    reciprocal_top3_bonus: float = 0.01,
    ambiguity_penalty: float = 0.005,
    endpoint_rank_weight: float = 0.015,
) -> np.ndarray:
    """Return a target-free ordering score from observed graph context."""
    base_score = np.asarray(scores, dtype=np.float64)
    if base_score.ndim != 1 or len(base_score) != len(features):
        raise ValueError("scores and graph features must be aligned")
    required = {
        "reciprocal_best",
        "reciprocal_top3",
        "ambiguity_log_degree",
        "min_endpoint_percentile",
        "category_score_percentile",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"graph features missing columns: {sorted(missing)}")

    out = features["category_score_percentile"].to_numpy(dtype=np.float64).copy()
    out += float(reciprocal_best_bonus) * features["reciprocal_best"].to_numpy(dtype=float)
    top3_only = (
        features["reciprocal_top3"].to_numpy(dtype=bool)
        & ~features["reciprocal_best"].to_numpy(dtype=bool)
    )
    out += float(reciprocal_top3_bonus) * top3_only.astype(float)
    out += float(endpoint_rank_weight) * features["min_endpoint_percentile"].to_numpy(float)
    out -= float(ambiguity_penalty) * features["ambiguity_log_degree"].to_numpy(float)
    if not np.isfinite(out).all():
        raise RuntimeError("graph rescore produced nonfinite values")
    return out


__all__ = ["graph_features", "graph_rescore"]
