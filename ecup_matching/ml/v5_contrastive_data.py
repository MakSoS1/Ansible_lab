from __future__ import annotations

import numpy as np
import pandas as pd


def select_fold_contrastive_pairs(
    frame: pd.DataFrame,
    fold_ids,
    base_oof_scores,
    *,
    held_fold: int,
    max_negative_to_positive: float = 2.0,
    hard_negative_fraction: float = 0.5,
    seed: int = 2026,
) -> pd.DataFrame:
    """Build a deterministic outer-train contrastive curriculum.

    Every positive from non-held development folds is retained. Negatives are
    capped relative to positives and split between high-scoring OOF hard
    negatives and random replay. Because the anchor scores are themselves OOF,
    hard-negative selection does not use in-sample predictions for those rows.
    """
    required = {"id1", "id2", "target", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    folds = np.asarray(fold_ids)
    base = np.asarray(base_oof_scores, dtype=np.float64)
    if not (len(frame) == len(folds) == len(base)):
        raise ValueError("frame, fold_ids and base_oof_scores must have equal length")
    unique = set(np.unique(folds).tolist())
    if held_fold not in unique:
        raise ValueError(f"held fold {held_fold!r} is not present")
    if max_negative_to_positive <= 0:
        raise ValueError("max_negative_to_positive must be positive")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be in [0,1]")
    if not np.isfinite(base).all():
        raise ValueError("base_oof_scores contain NaN or infinity")

    source = frame.reset_index(drop=True).copy()
    source["_source_row"] = np.arange(len(source), dtype=np.int64)
    source["_base_oof"] = base
    train = source.loc[folds != held_fold].copy()
    positives = train.loc[pd.to_numeric(train["target"], errors="raise") >= 0.5].copy()
    negatives = train.loc[pd.to_numeric(train["target"], errors="raise") < 0.5].copy()
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("outer training partition must contain positives and negatives")

    max_negatives = min(
        len(negatives),
        int(np.floor(len(positives) * float(max_negative_to_positive))),
    )
    max_negatives = max(1, max_negatives)
    hard_n = min(max_negatives, int(round(max_negatives * float(hard_negative_fraction))))

    hard = (
        negatives.sort_values(["_base_oof", "_source_row"], ascending=[False, True], kind="mergesort")
        .head(hard_n)
        .copy()
    )
    remaining_n = max_negatives - len(hard)
    if remaining_n:
        pool = negatives.loc[~negatives["_source_row"].isin(hard["_source_row"])].copy()
        random = pool.sample(n=min(remaining_n, len(pool)), random_state=int(seed) + int(held_fold))
        selected_negatives = pd.concat([hard, random], ignore_index=True)
    else:
        selected_negatives = hard

    selected = pd.concat([positives, selected_negatives], ignore_index=True)
    # Stable pseudo-random ordering avoids long same-label runs without changing
    # which examples were selected.
    rng = np.random.default_rng(int(seed) + 10_000 + int(held_fold))
    selected["_order"] = rng.random(len(selected))
    selected = selected.sort_values(["_order", "_source_row"], kind="mergesort").drop(columns="_order")
    if not selected["_source_row"].is_unique:
        raise RuntimeError("contrastive curriculum contains duplicate source rows")
    return selected.reset_index(drop=True)
