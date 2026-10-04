from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class TrainingReport:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    device: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_torch_surrogate(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
) -> TrainingReport:
    seed_everything(seed)
    dev = device or best_device()
    model.to(dev)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=dev)
    xv = torch.as_tensor(x_validation, dtype=torch.float32, device=dev)
    yv = torch.as_tensor(y_validation, dtype=torch.float32, device=dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()
    train_hist: list[float] = []
    val_hist: list[float] = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xt), yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_hist.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_hist.append(float(loss_fn(model(xv), yv).detach().cpu()))
    model.to("cpu")
    return TrainingReport(tuple(train_hist), tuple(val_hist), str(dev))
