from __future__ import annotations

import numpy as np
import pandas as pd

from .split import component_split


SEED = 2026
V1_VALID_FRACTION = 0.2


def fixed_v1_split(matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact deterministic outer split used by v1.

    Keeping this wrapper separate makes it difficult for later v2 code to
    silently change the headline validation protocol.
    """
    return component_split(matches, valid_fraction=V1_VALID_FRACTION, seed=SEED)


def calibration_split(
    matches: pd.DataFrame,
    outer_train_idx: np.ndarray,
    calibration_fraction: float = 0.125,
) -> tuple[np.ndarray, np.ndarray]:
    """Split only the outer-training components into fit/calibration rows.

    Returned indices always refer to the original ``matches`` frame.
    """
    outer = np.asarray(outer_train_idx, dtype=np.int64)
    if outer.ndim != 1 or len(outer) < 2:
        raise ValueError("outer_train_idx must contain at least two row indices")
    if len(np.unique(outer)) != len(outer):
        raise ValueError("outer_train_idx contains duplicate row indices")
    if np.any(outer < 0) or np.any(outer >= len(matches)):
        raise IndexError("outer_train_idx contains an out-of-range row index")

    subset = matches.iloc[outer].reset_index(drop=True)
    fit_local, calib_local = component_split(
        subset,
        valid_fraction=calibration_fraction,
        seed=SEED + 1,
    )
    return outer[fit_local], outer[calib_local]
