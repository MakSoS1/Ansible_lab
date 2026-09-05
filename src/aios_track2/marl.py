from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from .surrogates.graph import normalize_adjacency


class SharedGraphMAPPO(nn.Module):
    """Shared actor + centralized critic for well groups (CTDE)."""

    def __init__(self, obs_dim: int, action_dim: int, adjacency: torch.Tensor, hidden: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("adjacency", normalize_adjacency(adjacency))
        self.encoder = nn.Sequential(nn.Linear(obs_dim * 2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3 or obs.shape[1] != self.adjacency.shape[0]:
            raise ValueError("obs must be [batch,groups,features] matching adjacency")
        neigh = torch.einsum("ij,bjf->bif", self.adjacency, obs)
        return self.encoder(torch.cat([obs, neigh], dim=-1))

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        h = self.encode(obs)
        mean = self.actor_mean(h)
        std = self.log_std.clamp(-5, 1).exp().expand_as(mean)
        dist = Normal(mean, std)
        raw = mean if deterministic else dist.rsample()
        action = torch.tanh(raw)
        logp = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        logp = logp.sum(dim=-1)
        value = self.critic(h.mean(dim=1)).squeeze(-1)
        return action, logp, value

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        h = self.encode(obs)
        mean = self.actor_mean(h)
        std = self.log_std.clamp(-5, 1).exp().expand_as(mean)
        raw = torch.atanh(action.clamp(-0.999999, 0.999999))
        dist = Normal(mean, std)
        logp = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(h.mean(dim=1)).squeeze(-1)
        return logp, entropy, value

    def ppo_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        old_logp: torch.Tensor,
        advantage: torch.Tensor,
        returns: torch.Tensor,
        clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> torch.Tensor:
        logp, entropy, value = self.evaluate_actions(obs, action)
        group_adv = advantage[:, None].expand_as(logp)
        ratio = (logp - old_logp).exp()
        policy = -torch.min(ratio * group_adv, ratio.clamp(1 - clip, 1 + clip) * group_adv).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        return policy + value_coef * value_loss - entropy_coef * entropy.mean()
