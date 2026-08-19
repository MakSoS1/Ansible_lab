from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .metrics import macro_average_precision


def _validate_lengths(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    score = np.asarray(scores, dtype=float)
    if len(frame) != len(score):
        raise ValueError("frame and scores must have the same length")
    if not np.isfinite(score).all():
        raise ValueError("scores contain NaN or infinity")
    return score


def _stable_rank(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    # Random tie key is deterministic for the configured seed while preserving
    # strict score ordering for non-tied candidates.
    rng = np.random.default_rng(seed)
    out = frame.copy()
    out["_tie"] = rng.random(len(out))
    return out.sort_values(["_score", "_tie"], ascending=[False, True], kind="mergesort")


def select_hard_negatives(
    frame: pd.DataFrame,
    scores,
    *,
    count: int,
    priority_categories: set[str],
    priority_fraction: float = 0.50,
    seed: int = 2026,
) -> pd.DataFrame:
    """Select model-mined false positives with a reserved priority-category quota."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if not 0.0 <= priority_fraction <= 1.0:
        raise ValueError("priority_fraction must be between 0 and 1")
    if "target" not in frame.columns or "category" not in frame.columns:
        raise ValueError("frame must contain target and category")
    score = _validate_lengths(frame, scores)

    negative = frame.loc[frame["target"].astype(float) < 0.5].copy()
    if negative.empty or count == 0:
        return negative.iloc[:0].copy().reset_index(drop=True)
    negative["_score"] = score[frame["target"].astype(float).to_numpy() < 0.5]

    target_count = min(int(count), len(negative))
    priority_mask = negative["category"].astype(str).isin(priority_categories)
    priority = _stable_rank(negative.loc[priority_mask], seed)
    other = _stable_rank(negative.loc[~priority_mask], seed + 1)

    reserve = min(math.ceil(target_count * priority_fraction), len(priority))
    picked_priority = priority.head(reserve)
    remaining = target_count - len(picked_priority)

    # Fill from all not-yet-selected candidates by model score. This preserves
    # the reserved category capacity but does not artificially suppress a very
    # hard non-priority false positive once that quota is satisfied.
    selected_index = set(picked_priority.index)
    leftovers = negative.loc[~negative.index.isin(selected_index)].copy()
    leftovers = _stable_rank(leftovers, seed + 2)
    picked = pd.concat([picked_priority, leftovers.head(remaining)], axis=0)
    picked = picked.sort_values("_score", ascending=False, kind="mergesort")
    return picked.drop(columns=["_score", "_tie"], errors="ignore").reset_index(drop=True)


def select_best_blend(
    structured,
    neural,
    target,
    category,
    *,
    alphas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> dict[str, object]:
    """Choose neural blend weight using the exact competition Macro AP."""
    structured_score = np.asarray(structured, dtype=float)
    neural_score = np.asarray(neural, dtype=float)
    y_true = np.asarray(target, dtype=float)
    categories = np.asarray(category).astype(str)
    lengths = {len(structured_score), len(neural_score), len(y_true), len(categories)}
    if len(lengths) != 1:
        raise ValueError("structured, neural, target and category must have the same length")
    if len(y_true) == 0:
        raise ValueError("blend input must not be empty")
    if not np.isfinite(structured_score).all() or not np.isfinite(neural_score).all():
        raise ValueError("blend score contains NaN or infinity")
    if not alphas:
        raise ValueError("alphas must not be empty")
    for alpha in alphas:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("every alpha must be between 0 and 1")

    structured_macro, structured_per_category = macro_average_precision(
        y_true, structured_score, categories
    )
    candidates: list[dict[str, object]] = []
    for alpha_raw in alphas:
        alpha = float(alpha_raw)
        score = np.clip((1.0 - alpha) * structured_score + alpha * neural_score, 0.0, 1.0)
        macro, per_category = macro_average_precision(y_true, score, categories)
        candidates.append(
            {
                "alpha_neural": alpha,
                "macro_average_precision": float(macro),
                "per_category_ap": per_category,
                "scores": score,
            }
        )

    # Prefer less neural inference when AP ties exactly.
    best = max(candidates, key=lambda row: (float(row["macro_average_precision"]), -float(row["alpha_neural"])))
    return {
        **best,
        "structured_macro_average_precision": float(structured_macro),
        "structured_per_category_ap": structured_per_category,
    }
