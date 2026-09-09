from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score


def balanced_assignment(probs: np.ndarray, counts: tuple[int, int, int]) -> np.ndarray:
    """Maximum log-probability assignment under exact class counts."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probs must have shape (n, 3)")
    counts = tuple(int(c) for c in counts)
    if len(counts) != 3 or any(c < 0 for c in counts) or sum(counts) != len(probs):
        raise ValueError("counts must contain three non-negative values summing to n")

    slots = np.repeat(np.arange(3), np.asarray(counts, dtype=int))
    costs = -np.log(np.clip(probs[:, slots], 1e-12, 1.0))
    rows, cols = linear_sum_assignment(costs)
    pred = np.empty(len(probs), dtype=int)
    pred[rows] = slots[cols]
    return pred


def evaluate_count_constraint(probs: np.ndarray, y: np.ndarray) -> dict:
    """Compare unconstrained argmax with an oracle-count assignment on known labels."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    counts = tuple(np.bincount(y, minlength=3).astype(int).tolist())
    argmax_pred = probs.argmax(axis=1)
    constrained_pred = balanced_assignment(probs, counts)
    return {
        "argmax_accuracy": float(accuracy_score(y, argmax_pred)),
        "constrained_accuracy": float(accuracy_score(y, constrained_pred)),
        "argmax_counts": np.bincount(argmax_pred, minlength=3).astype(int).tolist(),
        "constrained_counts": np.bincount(constrained_pred, minlength=3).astype(int).tolist(),
        "target_counts": list(counts),
    }
