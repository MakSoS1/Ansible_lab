from __future__ import annotations

import numpy as np
import torch
from torch import nn

from aios_track2.surrogates.base import Prediction, ScenarioBatch, TrainingReport, evaluate_surrogate
from aios_track2.surrogates.tcn import TCNSurrogate


class GraphSAGE(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.self_lin = nn.Linear(channels, channels)
        self.neigh_lin = nn.Linear(channels, channels)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # nodes: B, T, W, C  adj: W, W
        neighbor = torch.einsum("ij,btjc->btic", adj, nodes)
        return torch.relu(self.self_lin(nodes) + self.neigh_lin(neighbor))


class GraphTemporalSurrogate(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 32,
        message_layers: int = 2,
        seed: int = 42,
        epochs: int = 8,
        dropout: float = 0.1,
        adjacency: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.epochs = epochs
        self.encoder = TCNSurrogate(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            seed=seed,
            epochs=1,
        )
        self.graphs = nn.ModuleList([GraphSAGE(in_channels) for _ in range(message_layers)])
        self.decoder = nn.Linear(in_channels, in_channels)
        n = 8 if adjacency is None else adjacency.shape[0]
        adj = adjacency if adjacency is not None else np.eye(n, dtype=np.float32)
        degree = adj.sum(axis=1, keepdims=True)
        normalized = adj / np.maximum(degree, 1.0)
        self.register_buffer("adj", torch.tensor(normalized, dtype=torch.float32))
        self.train_ids: tuple[str, ...] = ()
        self.report: TrainingReport | None = None

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, variance = self.encoder.forward(features)
        hidden = mean
        adj = self.adj
        if not isinstance(adj, torch.Tensor):
            adj = torch.as_tensor(adj)
        for layer in self.graphs:
            hidden = layer(hidden, adj[: hidden.shape[2], : hidden.shape[2]])
        decoded = self.decoder(hidden)
        field = decoded.mean(dim=2, keepdim=True).expand_as(decoded)
        mixed = 0.85 * decoded + 0.15 * field
        return mixed, variance + 0.05 * (mixed - mean).pow(2)

    def fit(self, train: ScenarioBatch, validation: ScenarioBatch) -> GraphTemporalSurrogate:
        self.train_ids = train.scenario_ids
        device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        x = torch.tensor(train.features, dtype=torch.float32, device=device)
        y = torch.tensor(train.targets, dtype=torch.float32, device=device)
        self.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            mean, _ = self.forward(x)
            loss = torch.nn.functional.huber_loss(mean, y) + 0.01 * torch.relu(-mean[..., 0]).mean()
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


class DeepEnsemble:
    def __init__(self, seeds: tuple[int, ...] = (11, 23, 42), **kwargs) -> None:
        self.seeds = seeds
        self.members = [GraphTemporalSurrogate(seed=seed, **kwargs) for seed in seeds]
        self.train_ids: tuple[str, ...] = ()
        self.report: TrainingReport | None = None

    def fit(self, train: ScenarioBatch, validation: ScenarioBatch) -> DeepEnsemble:
        reports = [member.fit(train, validation) for member in self.members]
        self.train_ids = train.scenario_ids
        self.report = reports[-1].report
        return self

    def predict(self, batch: ScenarioBatch) -> Prediction:
        means = []
        for member in self.members:
            means.append(member.predict(batch).mean)
        stacked = np.stack(means)
        return Prediction(mean=stacked.mean(axis=0), variance=np.maximum(stacked.var(axis=0), 0.0))
