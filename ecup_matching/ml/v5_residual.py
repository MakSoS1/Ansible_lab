from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _expit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    out = np.empty_like(value)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def clipped_logit(probability, *, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError("probability contains NaN or infinity")
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must be between 0 and 0.5")
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def apply_residual(base_score, residual, *, residual_strength: float) -> np.ndarray:
    base = np.asarray(base_score, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if base.shape != correction.shape:
        raise ValueError("base_score and residual must have the same shape")
    if not np.isfinite(correction).all():
        raise ValueError("residual contains NaN or infinity")
    if residual_strength < 0.0:
        raise ValueError("residual_strength must be non-negative")
    if residual_strength == 0.0 or not np.any(correction):
        return base.copy()
    return _expit(clipped_logit(base) + float(residual_strength) * correction)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)


@dataclass
class ResidualRanker:
    residual_strength: float = 0.25
    l2_regularization: float = 10.0
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.residual_strength < 0.0:
            raise ValueError("residual_strength must be non-negative")
        if self.l2_regularization <= 0.0:
            raise ValueError("l2_regularization must be positive")
        self.model_: Pipeline | None = None
        self.columns_: list[str] | None = None

    def _pipeline(self, frame: pd.DataFrame) -> Pipeline:
        categorical = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
        numeric = [c for c in frame.columns if c not in categorical]
        transformers = []
        if categorical:
            transformers.append(("categorical", _one_hot_encoder(), categorical))
        if numeric:
            transformers.append(("numeric", StandardScaler(), numeric))
        preprocess = ColumnTransformer(transformers, remainder="drop")
        regressor = Ridge(alpha=float(self.l2_regularization), random_state=self.seed)
        return Pipeline([("preprocess", preprocess), ("regressor", regressor)])

    def fit(self, features: pd.DataFrame, target, base_score, sample_weight=None) -> "ResidualRanker":
        if len(features) == 0:
            raise ValueError("features must not be empty")
        y = np.asarray(target, dtype=np.float64)
        base = np.asarray(base_score, dtype=np.float64)
        if not (len(features) == len(y) == len(base)):
            raise ValueError("features, target and base_score must have equal lengths")
        if ((y < 0.0) | (y > 1.0)).any():
            raise ValueError("target must be in [0,1]")

        # Learn only the probability-space error relative to the proven anchor.
        # The correction is later applied conservatively in logit space.
        residual_target = y - np.clip(base, 0.0, 1.0)
        self.columns_ = list(features.columns)
        self.model_ = self._pipeline(features)
        fit_kwargs = {}
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if len(weights) != len(features):
                raise ValueError("sample_weight must match features")
            fit_kwargs["regressor__sample_weight"] = weights
        self.model_.fit(features[self.columns_], residual_target, **fit_kwargs)
        return self

    def predict_residual(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.columns_ is None:
            raise RuntimeError("ResidualRanker is not fitted")
        missing = set(self.columns_) - set(features.columns)
        if missing:
            raise ValueError(f"features missing fitted columns: {sorted(missing)}")
        return np.asarray(self.model_.predict(features[self.columns_]), dtype=np.float64)

    def predict_proba(self, features: pd.DataFrame, base_score) -> np.ndarray:
        base = np.asarray(base_score, dtype=np.float64)
        if len(base) != len(features):
            raise ValueError("base_score must match features")
        correction = self.predict_residual(features)
        return apply_residual(base, correction, residual_strength=self.residual_strength)
