from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ConnectivityEdge:
    injector: str
    producer: str
    lag: int
    correlation: float


def compensation_ratio(*, injection: float, withdrawal: float) -> float:
    if withdrawal <= 0:
        return float("nan")
    return float(injection / withdrawal)


def detect_water_breakthrough(watercut: np.ndarray, *, jump_threshold: float = 0.15) -> int | None:
    values = np.asarray(watercut, dtype=float).reshape(-1)
    if len(values) < 2:
        return None
    jumps = np.diff(values)
    hits = np.flatnonzero(jumps >= jump_threshold)
    return int(hits[0] + 1) if len(hits) else None


def lagged_connectivity(
    injectors: dict[str, np.ndarray],
    producers: dict[str, np.ndarray],
    *,
    max_lag: int = 12,
    min_points: int = 12,
) -> dict[tuple[str, str], ConnectivityEdge]:
    result: dict[tuple[str, str], ConnectivityEdge] = {}
    for inj_name, inj_raw in sorted(injectors.items()):
        inj = np.asarray(inj_raw, dtype=float).reshape(-1)
        for prod_name, prod_raw in sorted(producers.items()):
            prod = np.asarray(prod_raw, dtype=float).reshape(-1)
            n = min(len(inj), len(prod))
            if n < min_points:
                continue
            best_lag, best_corr = 0, -np.inf
            for lag in range(0, min(max_lag, n - min_points) + 1):
                x = inj[: n - lag] if lag else inj[:n]
                y = prod[lag:n] if lag else prod[:n]
                finite = np.isfinite(x) & np.isfinite(y)
                if finite.sum() < min_points or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
                    continue
                corr = float(np.corrcoef(x[finite], y[finite])[0, 1])
                if (not np.isfinite(best_corr)) or abs(corr) > abs(best_corr):
                    best_lag, best_corr = lag, corr
            if not np.isfinite(best_corr):
                best_corr = 0.0
            result[(inj_name, prod_name)] = ConnectivityEdge(inj_name, prod_name, best_lag, best_corr)
    return result


def physical_consistency_metrics(
    frame: pd.DataFrame, *, max_bhp: float | None = None, max_wlpr: float = 500.0
) -> dict[str, float]:
    required = {"withdrawal", "injection", "BHP", "WCT", "WLPR"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing physical columns: {sorted(missing)}")
    withdrawal = frame["withdrawal"].to_numpy(float)
    injection = frame["injection"].to_numpy(float)
    ratios = np.divide(injection, withdrawal, out=np.full_like(injection, np.nan), where=withdrawal > 0)
    pressure_rate = 0.0 if max_bhp is None else float(np.mean(frame["BHP"].to_numpy(float) > max_bhp))
    wct = frame["WCT"].to_numpy(float)
    return {
        "compensation_mae_to_one": float(np.nanmean(np.abs(ratios - 1.0))),
        "pressure_violation_rate": pressure_rate,
        "watercut_violation_rate": float(np.mean((wct < 0) | (wct > 1))),
        "wlpr_violation_rate": float(np.mean(frame["WLPR"].to_numpy(float) > max_wlpr)),
    }
