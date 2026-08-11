from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .v5_fixed_blend import percentile_rank


FINAL_SIGNAL_NAMES: tuple[str, ...] = (
    "weak",
    "sparse",
    "explicit",
    "contrastive",
    "teacher",
    "typed_explicit",
)


def select_full_contrastive_pairs(
    frame: pd.DataFrame,
    base_oof_scores,
    *,
    max_negative_to_positive: float = 2.0,
    hard_negative_fraction: float = 0.5,
    seed: int = 2026,
) -> pd.DataFrame:
    """Build the full-development counterpart of the outer-fold contrastive curriculum."""
    required = {"id1", "id2", "target", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    base = np.asarray(base_oof_scores, dtype=np.float64)
    if len(base) != len(frame):
        raise ValueError("frame and base_oof_scores must have equal length")
    if not np.isfinite(base).all():
        raise ValueError("base_oof_scores contain NaN or infinity")
    if max_negative_to_positive <= 0:
        raise ValueError("max_negative_to_positive must be positive")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be in [0,1]")

    source = frame.reset_index(drop=True).copy()
    source["_source_row"] = np.arange(len(source), dtype=np.int64)
    source["_base_oof"] = base
    positives = source.loc[pd.to_numeric(source["target"], errors="raise") >= 0.5].copy()
    negatives = source.loc[pd.to_numeric(source["target"], errors="raise") < 0.5].copy()
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("development partition must contain positives and negatives")

    max_negatives = min(
        len(negatives),
        max(1, int(np.floor(len(positives) * float(max_negative_to_positive)))),
    )
    hard_n = min(max_negatives, int(round(max_negatives * float(hard_negative_fraction))))
    hard = (
        negatives.sort_values(["_base_oof", "_source_row"], ascending=[False, True], kind="mergesort")
        .head(hard_n)
        .copy()
    )
    remaining_n = max_negatives - len(hard)
    if remaining_n:
        pool = negatives.loc[~negatives["_source_row"].isin(hard["_source_row"])].copy()
        replay = pool.sample(n=min(remaining_n, len(pool)), random_state=int(seed))
        selected_negatives = pd.concat([hard, replay], ignore_index=True)
    else:
        selected_negatives = hard

    selected = pd.concat([positives, selected_negatives], ignore_index=True)
    rng = np.random.default_rng(int(seed) + 10_000)
    selected["_order"] = rng.random(len(selected))
    selected = selected.sort_values(["_order", "_source_row"], kind="mergesort")
    if not selected["_source_row"].is_unique:
        raise RuntimeError("contrastive curriculum contains duplicate source rows")
    return selected.drop(columns=["_order", "_base_oof", "_source_row"]).reset_index(drop=True)


def _ranked_final_signals(signals: Mapping[str, object]) -> tuple[dict[str, np.ndarray], int]:
    if set(signals) != set(FINAL_SIGNAL_NAMES):
        missing = sorted(set(FINAL_SIGNAL_NAMES) - set(signals))
        extra = sorted(set(signals) - set(FINAL_SIGNAL_NAMES))
        raise ValueError(f"signal set mismatch; missing={missing}, extra={extra}")
    ranked: dict[str, np.ndarray] = {}
    row_count: int | None = None
    for name in FINAL_SIGNAL_NAMES:
        values = np.asarray(signals[name], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"signal {name} must be finite and one-dimensional")
        if row_count is None:
            row_count = len(values)
        elif len(values) != row_count:
            raise ValueError("all signals must have equal length")
        ranked[name] = percentile_rank(values)
    if row_count is None or row_count == 0:
        raise ValueError("signals must not be empty")
    return ranked, row_count


def final_six_rank_fusion(signals: Mapping[str, object]) -> np.ndarray:
    """Exact target-free equal-rank fusion retained at strict OOF 0.5975445721."""
    ranked, _ = _ranked_final_signals(signals)
    return np.mean(np.vstack([ranked[name] for name in FINAL_SIGNAL_NAMES]), axis=0)


def _validated_simplex_weights(values, *, category: str) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (len(FINAL_SIGNAL_NAMES),):
        raise ValueError(
            f"category {category!r} weights must have length {len(FINAL_SIGNAL_NAMES)}"
        )
    if not np.isfinite(weights).all() or np.any(weights < -1e-12):
        raise ValueError(f"category {category!r} weights must be finite and nonnegative")
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"category {category!r} weights must sum to 1; observed={total}")
    return weights / total


def category_shrunk_six_rank_fusion(
    signals: Mapping[str, object],
    categories,
    model: Mapping[str, object],
) -> np.ndarray:
    """Apply the selected fixed category-shrunk simplex to target-free test ranks.

    The model is expected to be the deterministic full-development refit of the
    strict OOF-selected prior=8000 architecture. Unknown categories fail closed
    rather than silently falling back to a different scoring rule.
    """
    ranked, row_count = _ranked_final_signals(signals)
    category_array = np.asarray(categories).astype(str)
    if category_array.ndim != 1 or len(category_array) != row_count:
        raise ValueError("categories must be one-dimensional and aligned with signals")

    signal_names = tuple(str(value) for value in model.get("signal_names", ()))
    if signal_names != FINAL_SIGNAL_NAMES:
        raise ValueError(
            f"production signal order mismatch: {signal_names!r} != {FINAL_SIGNAL_NAMES!r}"
        )
    raw_category_weights = model.get("category_weights")
    if not isinstance(raw_category_weights, Mapping) or not raw_category_weights:
        raise ValueError("production category_weights must be a non-empty mapping")

    rank_matrix = np.column_stack([ranked[name] for name in FINAL_SIGNAL_NAMES])
    score = np.full(row_count, np.nan, dtype=np.float64)
    for category in sorted(np.unique(category_array).tolist()):
        if category not in raw_category_weights:
            raise ValueError(f"production model missing category {category!r}")
        weights = _validated_simplex_weights(
            raw_category_weights[category], category=category
        )
        mask = category_array == category
        score[mask] = rank_matrix[mask] @ weights
    if not np.isfinite(score).all():
        raise RuntimeError("category-shrunk production fusion failed to score every row")
    return np.clip(score, 0.0, 1.0)
