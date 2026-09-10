from __future__ import annotations

import numpy as np
import pandas as pd


def split_scenarios(
    frame: pd.DataFrame, *, seed: int = 42, train: float = 0.7, validation: float = 0.15
) -> dict[str, pd.DataFrame]:
    if "scenario_id" not in frame:
        raise ValueError("scenario_id column is required")
    if train <= 0 or validation <= 0 or train + validation >= 1:
        raise ValueError("split fractions must leave a positive test partition")
    ids = np.array(sorted(frame["scenario_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(n * train)))
    n_val = max(1, int(round(n * validation)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    partitions = {
        "train": set(ids[:n_train]),
        "validation": set(ids[n_train:n_train + n_val]),
        "test": set(ids[n_train + n_val:]),
    }
    return {
        name: frame[frame["scenario_id"].astype(str).isin(values)].copy().reset_index(drop=True)
        for name, values in partitions.items()
    }
