from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr


def align_common_post_start(date_arrays: Iterable[np.ndarray], *, start_date: str) -> np.ndarray:
    common: set[str] | None = None
    for values in date_arrays:
        current = {str(value) for value in np.asarray(values).astype(str) if str(value) >= start_date}
        common = current if common is None else common & current
    if common is None:
        raise ValueError("at least one date array is required")
    return np.asarray(sorted(common), dtype="U10")


def _r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=float).reshape(-1)
    p = np.asarray(prediction, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("truth and prediction must have the same shape")
    denominator = float(np.sum((y - y.mean()) ** 2))
    squared_error = float(np.sum((y - p) ** 2))
    if denominator <= 1e-15:
        return 1.0 if squared_error <= 1e-15 else float("nan")
    return 1.0 - squared_error / denominator


def dynamic_delta_report(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    baseline: np.ndarray,
    scenario_ids: Sequence[int],
    channels: Sequence[str],
) -> dict[str, Any]:
    y = np.asarray(truth, dtype=float)
    p = np.asarray(prediction, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if y.shape != p.shape or y.ndim != 3:
        raise ValueError("truth and prediction must have shape [scenario, time, channel]")
    if base.shape != y.shape[1:]:
        raise ValueError("baseline must have shape [time, channel]")
    if len(channels) != y.shape[2]:
        raise ValueError("channel names must match the final tensor dimension")
    ids = np.asarray(tuple(scenario_ids), dtype=int)
    if ids.size == 0:
        raise ValueError("at least one scenario id is required")

    truth_delta = y - base[None, :, :]
    prediction_delta = p - base[None, :, :]
    aggregate: dict[str, float] = {}
    cells: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(channels):
        aggregate[str(channel)] = _r2_score(
            truth_delta[ids, :, channel_index], prediction_delta[ids, :, channel_index]
        )
        for scenario_id in ids:
            value = _r2_score(
                truth_delta[scenario_id, :, channel_index], prediction_delta[scenario_id, :, channel_index]
            )
            if np.isfinite(value):
                cells.append({"scenario_id": int(scenario_id), "channel": str(channel), "r2": float(value)})
    if not cells:
        raise ValueError("no finite scenario-channel R2 values were available")
    cell_values = np.asarray([item["r2"] for item in cells], dtype=float)
    finite_aggregate = [value for value in aggregate.values() if np.isfinite(value)]
    if not finite_aggregate:
        raise ValueError("no finite aggregate channel R2 values were available")
    return {
        "aggregate_channel_r2": aggregate,
        "mean_aggregate_channel_r2": float(np.mean(finite_aggregate)),
        "min_aggregate_channel_r2": float(np.min(finite_aggregate)),
        "p10_scenario_channel_r2": float(np.quantile(cell_values, 0.10)),
        "worst_scenario_channel": min(cells, key=lambda item: item["r2"]),
        "evaluated_scenarios": [int(value) for value in ids],
    }


def ranking_report(truth_npv: np.ndarray, prediction_npv: np.ndarray, *, top_k: int = 3) -> dict[str, float]:
    truth = np.asarray(truth_npv, dtype=float).reshape(-1)
    prediction = np.asarray(prediction_npv, dtype=float).reshape(-1)
    if truth.shape != prediction.shape or truth.size < 2:
        raise ValueError("truth and prediction must be same-length vectors with at least two items")
    if not 1 <= top_k <= truth.size:
        raise ValueError("top_k must be within the number of candidates")

    correct = 0
    total = 0
    for left in range(truth.size):
        for right in range(left + 1, truth.size):
            true_delta = truth[left] - truth[right]
            if true_delta == 0.0:
                continue
            predicted_delta = prediction[left] - prediction[right]
            correct += int(np.sign(true_delta) == np.sign(predicted_delta))
            total += 1
    true_top = set(np.argsort(truth)[-top_k:].tolist())
    predicted_top = set(np.argsort(prediction)[-top_k:].tolist())
    selected = int(np.argmax(prediction))
    absolute_error = np.abs(truth - prediction)
    return {
        "spearman": float(spearmanr(truth, prediction).statistic),
        "pairwise_accuracy": float(correct / total) if total else 1.0,
        "top_k_recall": float(len(true_top & predicted_top) / top_k),
        "simple_regret": float(np.max(truth) - truth[selected]),
        "mae": float(np.mean(absolute_error)),
        "max_abs_error": float(np.max(absolute_error)),
    }


@dataclass
class RbfKernelRidge:
    length_scale: float
    ridge: float
    center: float = 1.0
    scale: float = 0.2
    _x_train: np.ndarray | None = None
    _coef: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.length_scale <= 0 or self.ridge < 0 or self.scale <= 0:
            raise ValueError("length_scale and scale must be positive and ridge non-negative")

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim != 2:
            raise ValueError("x must have shape [sample, feature]")
        return (values - self.center) / self.scale

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        a = self._normalize(left)
        b = self._normalize(right)
        distance_squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-0.5 * distance_squared / (self.length_scale**2))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RbfKernelRidge":
        inputs = np.asarray(x, dtype=float)
        targets = np.asarray(y, dtype=float)
        if inputs.ndim != 2 or targets.shape[0] != inputs.shape[0]:
            raise ValueError("x and y must share the sample dimension")
        train_kernel = self._kernel(inputs, inputs)
        system = train_kernel + self.ridge * np.eye(inputs.shape[0])
        self._x_train = inputs.copy()
        self._coef = np.linalg.solve(system, targets)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._x_train is None or self._coef is None:
            raise RuntimeError("fit must be called before predict")
        return self._kernel(np.asarray(x, dtype=float), self._x_train) @ self._coef
