from __future__ import annotations

import numpy as np


def compose_three_class_probs(
    p_dark: np.ndarray,
    p_bright_given_non_dark: np.ndarray,
) -> np.ndarray:
    """Compose P(class=0/1/2) from a dark gate and a 1-vs-2 gate.

    The second input is P(bright | non-dark), so P(normal | non-dark)
    is its complement.
    """
    p_dark = np.asarray(p_dark, dtype=float).reshape(-1)
    p_bright = np.asarray(p_bright_given_non_dark, dtype=float).reshape(-1)
    if p_dark.shape != p_bright.shape:
        raise ValueError("probability vectors must have the same shape")

    p_dark = np.clip(p_dark, 0.0, 1.0)
    p_bright = np.clip(p_bright, 0.0, 1.0)
    non_dark = 1.0 - p_dark
    out = np.column_stack([
        p_dark,
        non_dark * (1.0 - p_bright),
        non_dark * p_bright,
    ])
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out
