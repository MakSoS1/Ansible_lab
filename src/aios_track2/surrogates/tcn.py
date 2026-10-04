from __future__ import annotations

import torch
from torch import nn

from aios_track2.surrogates.base import Prediction, ScenarioBatch, TrainingReport, evaluate_surrogate


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        output = super().forward(inputs)
        pad = self.padding
        crop = int(pad[0]) if isinstance(pad, tuple) else int(pad)
        return output[:, :, :-crop] if crop else output


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.drop(self.act(self.conv(inputs)))


class TCNSurrogate(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        seed: int = 42,
        epochs: int = 12,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.input = nn.Conv1d(in_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *[TemporalBlock(hidden_channels, kernel_size, dilation, dropout) for dilation in dilations]
        )
        self.head = nn.Conv1d(hidden_channels, in_channels, 1)
        self.log_var = nn.Parameter(torch.zeros(1))
        self.train_ids: tuple[str, ...] = ()
        self.report: TrainingReport | None = None

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # features: B, T, W, C -> B*W, C, T
        batch, steps, wells, channels = features.shape
        reshaped = features.permute(0, 2, 3, 1).reshape(batch * wells, channels, steps)
        hidden = self.blocks(self.input(reshaped))
        mean = self.head(hidden).permute(0, 2, 1).reshape(batch, wells, steps, channels).permute(0, 2, 1, 3)
        variance = torch.exp(self.log_var).expand_as(mean)
        return mean, variance

    def fit(self, train: ScenarioBatch, validation: ScenarioBatch) -> TCNSurrogate:
        self.train_ids = train.scenario_ids
        device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        x = torch.tensor(train.features, dtype=torch.float32, device=device)
        y = torch.tensor(train.targets, dtype=torch.float32, device=device)
        self.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            mean, _ = self.forward(x)
            loss = torch.nn.functional.huber_loss(mean, y)
            loss.backward()
            optimizer.step()
        metrics = evaluate_surrogate(self, validation)
        self.report = TrainingReport(seed=self.seed, epochs=self.epochs, metrics=metrics, dataset_revision="local")
        return self

    def predict(self, batch: ScenarioBatch) -> Prediction:
        self.eval()
        with torch.no_grad():
            features = torch.tensor(batch.features, dtype=torch.float32, device=next(self.parameters()).device)
            mean, variance = self.forward(features)
        return Prediction(mean=mean.cpu().numpy(), variance=variance.cpu().numpy())
