from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aios_track2.surrogates.base import (
    Prediction,
    ScenarioBatch,
    TrainingReport,
    evaluate_surrogate,
)


@dataclass
class LinearSurrogate:
    seed: int = 42
    alpha: float = 1.0
    n_bootstrap: int = 8
    _weights: np.ndarray | None = None
    _bias: np.ndarray | None = None
    _boot_weights: np.ndarray | None = None
    _x_mean: np.ndarray = field(default_factory=lambda: np.zeros(1))
    _x_std: np.ndarray = field(default_factory=lambda: np.ones(1))
    _y_mean: np.ndarray = field(default_factory=lambda: np.zeros(1))
    _shape: tuple[int, ...] = ()
    _feature_names: tuple[str, ...] = ()
    train_ids: tuple[str, ...] = ()
    report: TrainingReport | None = None

    def fit(self, train: ScenarioBatch, validation: ScenarioBatch) -> LinearSurrogate:
        self.train_ids = train.scenario_ids
        x = train.features.reshape(train.features.shape[0], -1)
        y = train.targets.reshape(train.targets.shape[0], -1)
        x_mean = x.mean(axis=0)
        x_std = np.maximum(x.std(axis=0), 1e-6)
        y_mean = y.mean(axis=0)
        xs = (x - x_mean) / x_std
        ys = y - y_mean
        xtx = xs.T @ xs + self.alpha * np.eye(xs.shape[1])
        weights = np.linalg.solve(xtx, xs.T @ ys)
        rng = np.random.default_rng(self.seed)
        boots = []
        for _ in range(self.n_bootstrap):
            index = rng.integers(0, len(xs), size=len(xs))
            xtx_b = xs[index].T @ xs[index] + self.alpha * np.eye(xs.shape[1])
            boots.append(np.linalg.solve(xtx_b, xs[index].T @ ys[index]))
        self._weights = weights
        self._bias = np.concatenate([x_mean, x_std, y_mean])
        self._x_mean = x_mean
        self._x_std = x_std
        self._y_mean = y_mean
        self._boot_weights = np.stack(boots)
        self._shape = train.targets.shape[1:]
        self._feature_names = tuple(f"f{i}" for i in range(x.shape[1]))
        metrics = evaluate_surrogate(self, validation)
        self.report = TrainingReport(seed=self.seed, epochs=1, metrics=metrics, dataset_revision="local")
        return self

    def predict(self, batch: ScenarioBatch) -> Prediction:
        if self._weights is None or self._boot_weights is None:
            raise RuntimeError("model is not fit")
        x = batch.features.reshape(batch.features.shape[0], -1)
        xs = (x - self._x_mean) / self._x_std
        mean = xs @ self._weights + self._y_mean
        boot = np.stack([xs @ member + self._y_mean for member in self._boot_weights])
        variance = boot.var(axis=0)
        return Prediction(
            mean=mean.reshape((batch.features.shape[0], *self._shape)),
            variance=np.maximum(variance.reshape((batch.features.shape[0], *self._shape)), 0.0),
        )
