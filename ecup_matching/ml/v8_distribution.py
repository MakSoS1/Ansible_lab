from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _targets(frame: pd.DataFrame, target_col: str = "target") -> pd.Series:
    if target_col not in frame.columns:
        raise ValueError(f"frame missing target column {target_col!r}")
    target = pd.to_numeric(frame[target_col], errors="raise").astype(float)
    if len(target) == 0:
        raise ValueError("target frame must not be empty")
    values = target.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("target contains NaN or infinity")
    return target


def binary_prevalence(frame: pd.DataFrame, target_col: str = "target") -> float:
    target = _targets(frame, target_col)
    unique = set(pd.unique(target).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"binary prevalence requires only 0/1 targets; observed={sorted(unique)[:10]}")
    return float(target.mean())


def target_distribution(frame: pd.DataFrame, target_col: str = "target") -> dict[str, object]:
    target = _targets(frame, target_col)
    values = target.to_numpy(dtype=float)
    unique = set(pd.unique(target).tolist())
    is_binary = bool(unique.issubset({0.0, 1.0}))
    report: dict[str, object] = {
        "rows": int(len(target)),
        "mean": float(target.mean()),
        "std": float(target.std(ddof=1)) if len(target) > 1 else 0.0,
        "min": float(target.min()),
        "max": float(target.max()),
        "frac_eq_0": float(np.mean(values == 0.0)),
        "frac_eq_1": float(np.mean(values == 1.0)),
        "frac_lt_005": float(np.mean(values < 0.05)),
        "frac_gt_095": float(np.mean(values > 0.95)),
        "frac_020_080": float(np.mean((values >= 0.20) & (values <= 0.80))),
        "is_binary": is_binary,
        "quantiles": {
            str(q): float(target.quantile(q))
            for q in (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
        },
    }
    if is_binary:
        report["binary_prevalence"] = float(target.mean())
    return report


def candidate_graph_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    required = {"id1", "id2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"candidate frame missing columns: {sorted(missing)}")
    if len(frame) == 0:
        raise ValueError("candidate frame must not be empty")
    endpoints = pd.concat([frame["id1"], frame["id2"]], ignore_index=True)
    degree = endpoints.value_counts(sort=False)
    return {
        "rows": int(len(frame)),
        "unique_items": int(len(degree)),
        "mean_endpoint_degree": float(degree.mean()),
        "median_endpoint_degree": float(degree.median()),
        "p90_endpoint_degree": float(degree.quantile(0.90)),
        "p99_endpoint_degree": float(degree.quantile(0.99)),
        "max_endpoint_degree": int(degree.max()),
        "rows_per_unique_item": float(len(frame) / max(1, len(degree))),
    }


def human_excluded_llm_mask(llm_pairs: pd.DataFrame, human_ids: set[object]) -> np.ndarray:
    required = {"id1", "id2"}
    missing = required - set(llm_pairs.columns)
    if missing:
        raise ValueError(f"LLM pairs missing columns: {sorted(missing)}")
    left_human = llm_pairs["id1"].isin(human_ids).to_numpy(dtype=bool)
    right_human = llm_pairs["id2"].isin(human_ids).to_numpy(dtype=bool)
    return np.asarray(~(left_human | right_human), dtype=bool)


def category_distribution_report(
    pairs: pd.DataFrame,
    item_categories: Mapping[object, object],
    *,
    target_col: str = "target",
) -> dict[str, dict[str, object]]:
    required = {"id1", "id2", target_col}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs missing columns: {sorted(missing)}")
    left = pairs["id1"].map(item_categories)
    right = pairs["id2"].map(item_categories)
    if left.isna().any() or right.isna().any():
        raise KeyError("category mapping is missing at least one pair endpoint")
    mismatch = left.astype(str).to_numpy() != right.astype(str).to_numpy()
    if bool(np.any(mismatch)):
        raise ValueError("pair endpoints must belong to the same category")
    work = pairs.copy()
    work["_category"] = left.astype(str).to_numpy()
    report: dict[str, dict[str, object]] = {}
    for category, group in work.groupby("_category", sort=True):
        dist = target_distribution(group, target_col=target_col)
        # Keep the soft-target mean semantically distinct from true binary prevalence.
        dist["target_mean"] = dist.pop("mean")
        report[str(category)] = dist
    return report


__all__ = [
    "binary_prevalence",
    "candidate_graph_summary",
    "category_distribution_report",
    "human_excluded_llm_mask",
    "target_distribution",
]
