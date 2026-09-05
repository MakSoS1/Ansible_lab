from __future__ import annotations

import torch
from torch import nn


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    a = adjacency.float()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    a = torch.maximum(a, a.T) + torch.eye(a.shape[0], device=a.device)
    deg = a.sum(dim=1).clamp_min(1e-6)
    inv = deg.rsqrt()
    return inv[:, None] * a * inv[None, :]


class GraphTemporalSurrogate(nn.Module):
    def __init__(self, in_features: int, out_features: int, adjacency: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.register_buffer("adjacency", normalize_adjacency(adjacency))
        self.spatial = nn.Sequential(nn.Linear(in_features * 2, hidden), nn.GELU())
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("expected [batch,time,wells,features]")
        if x.shape[2] != self.adjacency.shape[0]:
            raise ValueError("well count differs from adjacency")
        neigh = torch.einsum("ij,btjf->btif", self.adjacency, x)
        h = self.spatial(torch.cat([x, neigh], dim=-1))
        b, t, w, d = h.shape
        seq = h.permute(0, 2, 1, 3).reshape(b * w, t, d)
        z, _ = self.temporal(seq)
        y = self.head(z).reshape(b, w, t, -1).permute(0, 2, 1, 3)
        return y
