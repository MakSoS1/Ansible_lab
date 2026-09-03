from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EnsemblePrediction:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_members(cls, members: list[np.ndarray]) -> "EnsemblePrediction":
        if not members:
            raise ValueError("ensemble must contain at least one prediction")
        stack = np.stack([np.asarray(m, dtype=float) for m in members], axis=0)
        return cls(stack.mean(axis=0), stack.std(axis=0, ddof=0))
