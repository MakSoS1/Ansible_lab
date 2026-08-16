from __future__ import annotations

import torch
from torch import nn


class V15PairResidualHead(nn.Module):
    """Zero-safe typed-feature residual over a frozen pair CrossEncoder logit."""

    def __init__(self, *, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_dim:
            raise ValueError(f"features must be [B,{self.feature_dim}]")
        return self.net(features).squeeze(-1)

    def forward(self, teacher_logit: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if teacher_logit.ndim != 1 or int(teacher_logit.shape[0]) != int(features.shape[0]):
            raise ValueError("teacher_logit must be [B] aligned with features")
        return teacher_logit + self.residual(features)
