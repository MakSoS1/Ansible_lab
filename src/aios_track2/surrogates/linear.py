from __future__ import annotations

import numpy as np

from .base import EnsemblePrediction


class LinearSurrogate:
    def __init__(self, ridge: float = 1e-4):
        self.ridge = float(ridge)
        self.coef_: np.ndarray | None = None
        self.resid_std_: np.ndarray | None = None
        self.output_shape_: tuple[int, ...] | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearSurrogate":
        x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x_arr.shape[:-1] != y_arr.shape[:-1]:
            raise ValueError("x and y must align except feature dimension")
        xf = x_arr.reshape(-1, x_arr.shape[-1])
        yf = y_arr.reshape(-1, y_arr.shape[-1])
        design = np.concatenate([xf, np.ones((len(xf), 1))], axis=1)
        reg = self.ridge * np.eye(design.shape[1])
        reg[-1, -1] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + reg, design.T @ yf)
        pred = design @ self.coef_
        self.resid_std_ = np.maximum(np.std(yf - pred, axis=0), 1e-8)
        self.output_shape_ = y_arr.shape[1:]
        return self

    def predict(self, x: np.ndarray) -> EnsemblePrediction:
        if self.coef_ is None or self.resid_std_ is None:
            raise RuntimeError("fit must be called before predict")
        x_arr = np.asarray(x, dtype=float)
        xf = x_arr.reshape(-1, x_arr.shape[-1])
        design = np.concatenate([xf, np.ones((len(xf), 1))], axis=1)
        out = design @ self.coef_
        mean = out.reshape(*x_arr.shape[:-1], out.shape[-1])
        std = np.broadcast_to(self.resid_std_, mean.shape).copy()
        return EnsemblePrediction(mean, std)
