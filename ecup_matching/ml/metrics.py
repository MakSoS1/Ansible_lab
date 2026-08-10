from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def macro_average_precision(
    y_true,
    y_score,
    categories,
) -> tuple[float, dict[str, float]]:
    """Compute the competition metric: mean sklearn average precision over categories."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    categories = np.asarray(categories).astype(str)

    if not (len(y_true) == len(y_score) == len(categories)):
        raise ValueError("y_true, y_score and categories must have equal lengths")
    if len(y_true) == 0:
        raise ValueError("metric input must not be empty")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains NaN or infinity")

    per_category: dict[str, float] = {}
    for category in sorted(np.unique(categories).tolist()):
        mask = categories == category
        per_category[category] = float(average_precision_score(y_true[mask], y_score[mask]))

    return float(np.mean(list(per_category.values()))), per_category
