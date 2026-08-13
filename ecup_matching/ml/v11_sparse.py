from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SparseConfig:
    n_features: int = 65536
    batch_size: int = 50000
