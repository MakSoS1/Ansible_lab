from __future__ import annotations

import numpy as np
import pandas as pd

from .label_graph import canonicalize_pairs, positive_components


def weak_confidence_weight(probability: float) -> float:
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("weak target must be in [0,1]")
    if p <= 0.03 or p >= 0.97:
        return 1.0
    if p <= 0.15 or p >= 0.85:
        return 0.6
    if p <= 0.30 or p >= 0.70:
        return 0.3
    return 0.0


def prepare_weak_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = {"id1", "id2", "target"} - set(df.columns)
    if missing:
        raise ValueError(f"weak pairs missing required columns: {sorted(missing)}")
    out = canonicalize_pairs(df).reset_index(drop=True)
    target = pd.to_numeric(out["target"], errors="raise").astype(float)
    if ((target < 0.0) | (target > 1.0)).any():
        raise ValueError("weak target must be in [0,1]")
    out["weak_weight"] = target.map(weak_confidence_weight).astype(float)
    out["hard_target"] = (target >= 0.5).astype(np.int8)
    excluded = int((out["weak_weight"] <= 0).sum())
    out = out[out["weak_weight"] > 0].reset_index(drop=True)
    # Collapse exact duplicated weak pairs deterministically, preferring the
    # highest-confidence target. This prevents frequent pseudo-labels from
    # accidentally becoming extra sample weight.
    if len(out):
        out["_confidence"] = (out["target"].astype(float) - 0.5).abs()
        out = (
            out.sort_values(["id1", "id2", "_confidence"], ascending=[True, True, False], kind="mergesort")
            .drop_duplicates(["id1", "id2"], keep="first")
            .drop(columns="_confidence")
            .reset_index(drop=True)
        )
    return out, {
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "excluded_mid_confidence": excluded,
    }


def remove_human_conflicts(
    weak: pd.DataFrame,
    human: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove weak examples that would overwrite authoritative human identity facts."""
    w = canonicalize_pairs(weak).reset_index(drop=True)
    h = canonicalize_pairs(human).reset_index(drop=True)
    human_pairs = set(h[["id1", "id2"]].itertuples(index=False, name=None))
    components = positive_components(h)

    exact_mask = np.fromiter(
        ((a, b) in human_pairs for a, b in w[["id1", "id2"]].itertuples(index=False, name=None)),
        dtype=bool,
        count=len(w),
    )
    remaining = w.loc[~exact_mask].copy()

    false_negative_mask: list[bool] = []
    for row in remaining.itertuples(index=False):
        a, b = row.id1, row.id2
        hard = getattr(row, "hard_target", int(float(row.target) >= 0.5))
        same_positive_component = a in components and b in components and components[a] == components[b]
        false_negative_mask.append(bool(hard == 0 and same_positive_component))

    false_negative = np.asarray(false_negative_mask, dtype=bool)
    out = remaining.loc[~false_negative].reset_index(drop=True)
    return out, {
        "input_rows": int(len(w)),
        "output_rows": int(len(out)),
        "exact_human_pairs_removed": int(exact_mask.sum()),
        "component_false_negatives_removed": int(false_negative.sum()),
    }


def sample_weak_training(
    weak: pd.DataFrame,
    max_rows: int,
    seed: int = 2026,
    category_column: str = "category",
) -> pd.DataFrame:
    """Deterministically cap weak data while approximately balancing categories/classes."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if len(weak) <= max_rows:
        return weak.copy().reset_index(drop=True)
    if category_column not in weak.columns:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(weak), size=max_rows, replace=False))
        return weak.iloc[idx].reset_index(drop=True)

    frame = weak.copy().reset_index(drop=True)
    if "hard_target" not in frame.columns:
        frame["hard_target"] = (frame["target"].astype(float) >= 0.5).astype(np.int8)
    groups = list(frame.groupby([category_column, "hard_target"], sort=True, dropna=False))
    quota = max(1, max_rows // max(1, len(groups)))
    selected: list[pd.DataFrame] = []
    used: set[int] = set()
    for group_number, (_, group) in enumerate(groups):
        take = min(quota, len(group))
        sample = group.sample(n=take, random_state=seed + group_number)
        selected.append(sample)
        used.update(sample.index.tolist())
    chosen = pd.concat(selected, axis=0) if selected else frame.iloc[:0]
    remaining_n = max_rows - len(chosen)
    if remaining_n > 0:
        pool = frame.loc[~frame.index.isin(used)]
        if len(pool):
            chosen = pd.concat(
                [chosen, pool.sample(n=min(remaining_n, len(pool)), random_state=seed + 991)],
                axis=0,
            )
    return chosen.sort_index(kind="mergesort").head(max_rows).reset_index(drop=True)
