from __future__ import annotations

import torch
from torch import nn


class TCNSurrogate(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, hidden, kernel_size=3, padding=2, dilation=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=2),
            nn.GELU(),
        )
        self.head = nn.Conv1d(hidden, out_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("expected [batch,time,wells,features]")
        b, t, w, f = x.shape
        z = x.permute(0, 2, 3, 1).reshape(b * w, f, t)
        h = self.net(z)[..., :t]
        y = self.head(h).reshape(b, w, -1, t).permute(0, 3, 1, 2)
        return y
