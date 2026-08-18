from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .label_graph import canonicalize_pairs


def continuous_weak_weight(
    probability: float,
    dead_zone: float = 0.05,
    gamma: float = 1.5,
) -> float:
    """Continuous confidence for a soft weak target.

    The historical path removed the complete 0.30-0.70 band and quantized the
    rest into three weights. v18 keeps a narrow uncertainty dead zone and lets
    confidence increase smoothly with distance from 0.5.
    """
    p = float(probability)
    d = float(dead_zone)
    g = float(gamma)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("weak target must be finite and in [0,1]")
    if not math.isfinite(d) or not 0.0 <= d < 0.5:
        raise ValueError("dead_zone must be in [0,0.5)")
    if not math.isfinite(g) or g <= 0.0:
        raise ValueError("gamma must be finite and positive")
    margin = abs(p - 0.5)
    if margin <= d + 1e-12:
        return 0.0
    scaled = (margin - d) / (0.5 - d)
    return float(min(1.0, max(0.05, scaled**g)))


def prepare_weak_pairs_v18(
    df: pd.DataFrame,
    *,
    dead_zone: float = 0.05,
    gamma: float = 1.5,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    missing = {"id1", "id2", "target"} - set(df.columns)
    if missing:
        raise ValueError(f"weak pairs missing required columns: {sorted(missing)}")
    out = canonicalize_pairs(df).reset_index(drop=True)
    target = pd.to_numeric(out["target"], errors="raise").astype(float)
    if (~np.isfinite(target.to_numpy(float))).any() or ((target < 0.0) | (target > 1.0)).any():
        raise ValueError("weak target must be finite and in [0,1]")
    out["target"] = target
    out["_weak_margin"] = (target - 0.5).abs().astype(float)
    out["weak_weight"] = target.map(
        lambda value: continuous_weak_weight(value, dead_zone=dead_zone, gamma=gamma)
    ).astype(float)
    out["hard_target"] = (target >= 0.5).astype(np.int8)

    dead_zone_removed = int((out["weak_weight"] <= 0.0).sum())
    out = out[out["weak_weight"] > 0.0].copy().reset_index(drop=True)

    before_dedup = len(out)
    if len(out):
        # Prefer the target furthest from 0.5 for duplicate canonical pairs;
        # mergesort makes ties deterministic with original order as tie-break.
        out = (
            out.sort_values(
                ["id1", "id2", "_weak_margin"],
                ascending=[True, True, False],
                kind="mergesort",
            )
            .drop_duplicates(["id1", "id2"], keep="first")
            .reset_index(drop=True)
        )
    duplicate_rows_removed = int(before_dedup - len(out))

    report: dict[str, int | float] = {
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "dead_zone_removed": dead_zone_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "dead_zone": float(dead_zone),
        "gamma": float(gamma),
    }
    return out, report


def split_weak_curriculum(
    frame: pd.DataFrame,
    *,
    high_margin: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if "_weak_margin" not in frame.columns:
        raise ValueError("v18 weak frame requires _weak_margin")
    margin = pd.to_numeric(frame["_weak_margin"], errors="raise").astype(float)
    if not math.isfinite(float(high_margin)) or not 0.0 <= float(high_margin) <= 0.5:
        raise ValueError("high_margin must be in [0,0.5]")
    broad = frame.copy().reset_index(drop=True)
    high = broad[margin.to_numpy(float) >= float(high_margin) - 1e-12].copy().reset_index(drop=True)
    if len(broad) and not len(high):
        raise ValueError("high-confidence curriculum phase is empty")
    return high, broad, {"high_rows": int(len(high)), "broad_rows": int(len(broad))}


__all__ = [
    "continuous_weak_weight",
    "prepare_weak_pairs_v18",
    "split_weak_curriculum",
]
