from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
