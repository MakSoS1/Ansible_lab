from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def generate_sobol_trajectories(
    scenarios: int, steps: int, dimensions: int, *, seed: int = 42, max_delta: float = 0.15
) -> np.ndarray:
    if min(scenarios, steps, dimensions) <= 0:
        raise ValueError("scenarios, steps and dimensions must be positive")
    if not 0 < max_delta <= 1:
        raise ValueError("max_delta must be in (0, 1]")
    sampler = qmc.Sobol(d=steps * dimensions, scramble=True, seed=seed)
    raw = sampler.random(scenarios).reshape(scenarios, steps, dimensions)
    out = raw.copy()
    for t in range(1, steps):
        lo = np.maximum(0.0, out[:, t - 1] - max_delta)
        hi = np.minimum(1.0, out[:, t - 1] + max_delta)
        out[:, t] = np.clip(out[:, t], lo, hi)
    return out
