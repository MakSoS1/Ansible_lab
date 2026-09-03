from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def evaluate_surrogate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError("prediction and target shapes differ")
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    scale = float(np.nanmax(y) - np.nanmin(y))
    nrmse = rmse / max(scale, 1e-12)
    smape = float(np.mean(2 * np.abs(err) / np.maximum(np.abs(y) + np.abs(p), 1e-12)))
    centered = y - np.mean(y)
    total_variance = float(np.sum(centered**2))
    r2 = 1.0 - float(np.sum(err**2)) / max(total_variance, 1e-12)
    return {"mae": mae, "rmse": rmse, "nrmse": nrmse, "smape": smape, "r2": r2}


def ranking_metrics(true_npv: np.ndarray, predicted_npv: np.ndarray, *, top_k: int = 10) -> dict[str, float]:
    true = np.asarray(true_npv, dtype=float).reshape(-1)
    pred = np.asarray(predicted_npv, dtype=float).reshape(-1)
    if len(true) != len(pred) or len(true) < 2:
        raise ValueError("ranking metrics require equal arrays of length >= 2")
    k = min(max(1, top_k), len(true))
    true_top = set(np.argsort(true)[-k:])
    pred_top = set(np.argsort(pred)[-k:])
    overlap = len(true_top & pred_top)
    return {
        "spearman": float(spearmanr(true, pred).statistic),
        "kendall": float(kendalltau(true, pred).statistic),
        "top_k_recall": overlap / k,
        "pairwise_accuracy": _pairwise_accuracy(true, pred),
    }


def _pairwise_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    correct = total = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0:
                continue
            total += 1
            correct += int(da == db)
    return correct / max(total, 1)


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y_true)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def rollout_error_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.ndim < 2:
        raise ValueError("rollout arrays must share shape and have time axis=1")
    axes = tuple(i for i in range(y.ndim) if i != 1)
    return np.mean(np.abs(p - y), axis=axes)
