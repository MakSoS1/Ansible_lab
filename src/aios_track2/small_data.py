from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _quadratic_features(x: np.ndarray, *, center: float = 1.0) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 2:
        raise ValueError("x must have shape [sample, feature]")
    z = values - float(center)
    columns = [np.ones(values.shape[0], dtype=float)]
    columns.extend(z[:, index] for index in range(z.shape[1]))
    for left in range(z.shape[1]):
        for right in range(left, z.shape[1]):
            columns.append(z[:, left] * z[:, right])
    return np.stack(columns, axis=1)


@dataclass
class QuadraticRidge:
    ridge: float = 1e-6
    center: float = 1.0
    _coef: np.ndarray | None = None
    _target_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "QuadraticRidge":
        features = _quadratic_features(x, center=self.center)
        targets = np.asarray(y, dtype=float)
        if targets.shape[0] != features.shape[0]:
            raise ValueError("x and y must share the sample dimension")
        self._target_shape = targets.shape[1:]
        flat = targets.reshape(targets.shape[0], -1)
        regularizer = self.ridge * np.eye(features.shape[1])
        regularizer[0, 0] = 0.0
        system = features.T @ features + regularizer
        self._coef = np.linalg.pinv(system, rcond=1e-12) @ features.T @ flat
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("fit must be called before predict")
        features = _quadratic_features(x, center=self.center)
        flat = features @ self._coef
        return flat.reshape((features.shape[0],) + self._target_shape)


@dataclass
class StationaryKernelRidge:
    kind: str = "rbf"
    length_scale: float = 2.0
    ridge: float = 1e-4
    center: float = 1.0
    scale: float = 0.2
    _x_train: np.ndarray | None = None
    _coef: np.ndarray | None = None
    _target_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"rbf", "matern52"}:
            raise ValueError("kind must be 'rbf' or 'matern52'")
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
        if self.kind == "rbf":
            return np.exp(-0.5 * distance_squared / (self.length_scale**2))
        distance = np.sqrt(np.maximum(distance_squared, 0.0))
        q = np.sqrt(5.0) * distance / self.length_scale
        return (1.0 + q + q**2 / 3.0) * np.exp(-q)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "StationaryKernelRidge":
        inputs = np.asarray(x, dtype=float)
        targets = np.asarray(y, dtype=float)
        if inputs.ndim != 2 or targets.shape[0] != inputs.shape[0]:
            raise ValueError("x and y must share the sample dimension")
        self._target_shape = targets.shape[1:]
        flat = targets.reshape(targets.shape[0], -1)
        kernel = self._kernel(inputs, inputs)
        system = kernel + self.ridge * np.eye(inputs.shape[0])
        self._x_train = inputs.copy()
        self._coef = np.linalg.solve(system, flat)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._x_train is None or self._coef is None:
            raise RuntimeError("fit must be called before predict")
        values = np.asarray(x, dtype=float)
        flat = self._kernel(values, self._x_train) @ self._coef
        return flat.reshape((values.shape[0],) + self._target_shape)


def project_temporal_policy(
    values: np.ndarray,
    *,
    groups: int,
    nodes: int,
    lower: float,
    upper: float,
    max_delta: float,
) -> np.ndarray:
    if min(groups, nodes) <= 0:
        raise ValueError("groups and nodes must be positive")
    if not lower < upper:
        raise ValueError("lower must be smaller than upper")
    if not 0 < max_delta <= upper - lower:
        raise ValueError("max_delta must be positive and no wider than the policy range")
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != groups * nodes:
        raise ValueError("values must have shape [sample, groups * nodes]")
    matrix = np.clip(array, lower, upper).reshape(array.shape[0], groups, nodes).copy()
    for node in range(1, nodes):
        previous = matrix[:, :, node - 1]
        matrix[:, :, node] = np.clip(matrix[:, :, node], previous - max_delta, previous + max_delta)
        matrix[:, :, node] = np.clip(matrix[:, :, node], lower, upper)
    return matrix.reshape(array.shape)
