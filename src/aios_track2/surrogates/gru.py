from __future__ import annotations

import torch
from torch import nn


class GRUSurrogate(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("expected [batch,time,wells,features]")
        b, t, w, f = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b * w, t, f)
        h, _ = self.gru(z)
        y = self.head(h).reshape(b, w, t, -1).permute(0, 2, 1, 3)
        return y
