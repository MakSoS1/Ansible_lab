from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np


@dataclass
class ScenarioBatch:
    scenario_ids: tuple[str, ...]
    features: np.ndarray
    targets: np.ndarray
    controls: np.ndarray

    def permute_wells(self, seed: int) -> ScenarioBatch:
        rng = np.random.default_rng(seed)
        order = rng.permutation(self.targets.shape[2])
        return ScenarioBatch(
            scenario_ids=self.scenario_ids,
            features=self.features[:, :, order, :],
            targets=self.targets[:, :, order, :],
            controls=self.controls[:, :, order, :],
        )


@dataclass
class Prediction:
    mean: np.ndarray
    variance: np.ndarray


@dataclass
class SurrogateMetrics:
    mae: dict[str, float]
    nrmse: dict[str, float]
    npv_mae: float
    spearman: float


@dataclass
class TrainingReport:
    seed: int
    epochs: int
    metrics: SurrogateMetrics
    dataset_revision: str


class SurrogateProtocol(Protocol):
    def fit(self, train: ScenarioBatch, validation: ScenarioBatch) -> Self: ...

    def predict(self, batch: ScenarioBatch) -> Prediction: ...


def evaluate_surrogate(model: SurrogateProtocol, test: ScenarioBatch, train_ids: tuple[str, ...] | None = None) -> SurrogateMetrics:
    if train_ids is not None and set(test.scenario_ids) & set(train_ids):
        raise ValueError("scenario leakage: test ids overlap train")
    prediction = model.predict(test)
    residual = np.abs(prediction.mean - test.targets)
    mae = float(residual.mean())
    scale = np.maximum(np.std(test.targets), 1e-6)
    nrmse = float(residual.mean() / scale)
    pred_npv = prediction.mean[..., 0].sum(axis=(1, 2))
    true_npv = test.targets[..., 0].sum(axis=(1, 2))
    spearman = _spearman(pred_npv, true_npv)
    return SurrogateMetrics(
        mae={"oil": mae},
        nrmse={"oil": nrmse},
        npv_mae=float(np.abs(pred_npv - true_npv).mean()),
        spearman=spearman,
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(values))
        return ranks

    if len(left) < 2:
        return 1.0
    a = rank(left)
    b = rank(right)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def is_ood(prediction: Prediction, threshold: float) -> np.ndarray:
    score = prediction.variance.mean(axis=tuple(range(1, prediction.variance.ndim)))
    return score > threshold
