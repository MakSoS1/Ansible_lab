from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.model_selection import train_test_split


def replay_balanced_test_labels(
    seed: int,
    *,
    per_class_total: int = 600,
    test_size: int = 300,
    n_classes: int = 3,
) -> np.ndarray:
    """Replay sklearn's stratified train/test split and return labels in test order.

    This models a dataset generator that starts with contiguous class blocks and
    calls ``train_test_split(..., stratify=y, random_state=seed)``.  Only the
    returned test-order labels are used; no hidden labels are read.
    """
    if per_class_total <= 0:
        raise ValueError("per_class_total must be positive")
    if n_classes < 2:
        raise ValueError("n_classes must be at least two")
    total = int(per_class_total) * int(n_classes)
    if not 0 < int(test_size) < total:
        raise ValueError("test_size must be between zero and total size")

    y = np.repeat(np.arange(n_classes, dtype=np.int64), per_class_total)
    indices = np.arange(total, dtype=np.int64)
    _, test_idx = train_test_split(
        indices,
        test_size=int(test_size),
        random_state=int(seed),
        stratify=y,
    )
    return y[test_idx]


def _coerce_proxies(proxy_predictions) -> list[np.ndarray]:
    if isinstance(proxy_predictions, np.ndarray):
        proxies = [proxy_predictions]
    else:
        proxies = list(proxy_predictions)
    if not proxies:
        raise ValueError("at least one proxy prediction vector is required")

    result: list[np.ndarray] = []
    expected_len = None
    for proxy in proxies:
        arr = np.asarray(proxy, dtype=np.int64).reshape(-1)
        if expected_len is None:
            expected_len = len(arr)
        elif len(arr) != expected_len:
            raise ValueError("all proxy vectors must have the same length")
        result.append(arr)
    return result


def rank_candidate_seeds(
    proxy_predictions,
    *,
    seeds: Iterable[int],
    per_class_total: int = 600,
    test_size: int = 300,
    n_classes: int = 3,
) -> list[dict]:
    """Rank split seeds by label agreement with one or more proxy predictions."""
    proxies = _coerce_proxies(proxy_predictions)
    rows: list[dict] = []

    for seed in seeds:
        candidate = replay_balanced_test_labels(
            int(seed),
            per_class_total=per_class_total,
            test_size=test_size,
            n_classes=n_classes,
        )
        if any(len(proxy) != len(candidate) for proxy in proxies):
            raise ValueError("proxy length does not match replayed test size")

        agreements = [float(np.mean(candidate == proxy)) for proxy in proxies]
        mean_agreement = float(np.mean(agreements))
        row = {
            "seed": int(seed),
            "agreement": agreements[0],
            "agreements": agreements,
            "mean_agreement": mean_agreement,
        }
        rows.append(row)

    rows.sort(key=lambda row: (-row["mean_agreement"], row["seed"]))
    return rows
