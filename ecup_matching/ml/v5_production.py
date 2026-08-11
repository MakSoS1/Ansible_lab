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


def final_six_rank_fusion(signals: Mapping[str, object]) -> np.ndarray:
    """Exact target-free equal-rank fusion retained at strict OOF 0.5975445721."""
    if set(signals) != set(FINAL_SIGNAL_NAMES):
        missing = sorted(set(FINAL_SIGNAL_NAMES) - set(signals))
        extra = sorted(set(signals) - set(FINAL_SIGNAL_NAMES))
        raise ValueError(f"signal set mismatch; missing={missing}, extra={extra}")
    ranked: list[np.ndarray] = []
    row_count: int | None = None
    for name in FINAL_SIGNAL_NAMES:
        values = np.asarray(signals[name], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"signal {name} must be finite and one-dimensional")
        if row_count is None:
            row_count = len(values)
        elif len(values) != row_count:
            raise ValueError("all signals must have equal length")
        ranked.append(percentile_rank(values))
    if row_count == 0:
        raise ValueError("signals must not be empty")
    return np.mean(np.vstack(ranked), axis=0)
