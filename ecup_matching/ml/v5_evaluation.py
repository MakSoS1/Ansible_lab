from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .metrics import OFFICIAL_CATEGORIES, macro_average_precision


def macro_ap_report(
    frame: pd.DataFrame,
    scores,
    *,
    category_col: str = "category",
    target_col: str = "target",
    strict_official: bool = False,
) -> dict[str, Any]:
    if target_col not in frame.columns or category_col not in frame.columns:
        raise ValueError(f"frame must contain {target_col!r} and {category_col!r}")
    score = np.asarray(scores, dtype=float)
    if len(score) != len(frame):
        raise ValueError("scores must have the same length as frame")
    macro, per_category = macro_average_precision(
        frame[target_col].to_numpy(),
        score,
        frame[category_col].astype(str).to_numpy(),
        expected_categories=OFFICIAL_CATEGORIES if strict_official else None,
        require_both_classes=bool(strict_official),
    )
    return {
        "rows": int(len(frame)),
        "macro_average_precision": float(macro),
        "per_category_ap": {str(k): float(v) for k, v in per_category.items()},
    }


def candidate_freeze_hash(
    config: dict[str, Any],
    *,
    prediction_sha: str,
    split_sha: str,
) -> str:
    payload = {
        "config": config,
        "prediction_sha": str(prediction_sha),
        "split_sha": str(split_sha),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def assert_gold_evaluation_eligible(
    freeze: dict[str, Any],
    *,
    split_sha: str,
    config_sha: str,
    prediction_sha: str,
) -> None:
    if not bool(freeze.get("frozen")):
        raise ValueError("candidate must be frozen before gold evaluation")
    expected = {
        "split_sha": str(split_sha),
        "config_sha": str(config_sha),
        "prediction_sha": str(prediction_sha),
    }
    for key, value in expected.items():
        if str(freeze.get(key, "")) != value:
            raise ValueError(f"gold evaluation {key} does not match frozen candidate")


def paired_component_bootstrap(
    frame: pd.DataFrame,
    base_scores,
    candidate_scores,
    *,
    component_col: str,
    category_col: str = "category",
    target_col: str = "target",
    n_bootstrap: int = 1000,
    seed: int = 2026,
) -> dict[str, float | int]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if component_col not in frame.columns:
        raise ValueError(f"frame missing component column {component_col!r}")
    base = np.asarray(base_scores, dtype=float)
    candidate = np.asarray(candidate_scores, dtype=float)
    if not (len(frame) == len(base) == len(candidate)):
        raise ValueError("frame and score arrays must have equal length")

    point_base = macro_ap_report(
        frame,
        base,
        category_col=category_col,
        target_col=target_col,
    )["macro_average_precision"]
    point_candidate = macro_ap_report(
        frame,
        candidate,
        category_col=category_col,
        target_col=target_col,
    )["macro_average_precision"]

    groups: dict[object, np.ndarray] = {}
    for component, indices in frame.groupby(component_col, sort=False).indices.items():
        groups[component] = np.asarray(indices, dtype=np.int64)
    components = list(groups)
    if len(components) < 2:
        raise ValueError("bootstrap requires at least two components")

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for iteration in range(n_bootstrap):
        sampled = rng.integers(0, len(components), size=len(components))
        row_indices = np.concatenate([groups[components[int(i)]] for i in sampled])
        sampled_frame = frame.iloc[row_indices].reset_index(drop=True)
        sampled_base = base[row_indices]
        sampled_candidate = candidate[row_indices]
        base_ap = macro_ap_report(
            sampled_frame,
            sampled_base,
            category_col=category_col,
            target_col=target_col,
        )["macro_average_precision"]
        candidate_ap = macro_ap_report(
            sampled_frame,
            sampled_candidate,
            category_col=category_col,
            target_col=target_col,
        )["macro_average_precision"]
        deltas[iteration] = float(candidate_ap - base_ap)

    return {
        "components": int(len(components)),
        "bootstrap_samples": int(n_bootstrap),
        "point_base": float(point_base),
        "point_candidate": float(point_candidate),
        "point_delta": float(point_candidate - point_base),
        "median_delta": float(np.median(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }
