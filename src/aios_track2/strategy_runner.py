from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .marl import SharedGraphMAPPO
from .optimization import Objective, diagonal_cma_es, risk_aware_cem


@dataclass(frozen=True, slots=True)
class StrategyRun:
    strategy: str
    x: np.ndarray
    score: float
    uncertainty: float
    valid: bool
    evaluations: int


def _local_search(objective: Objective, dim: int, seed: int, budget: int) -> StrategyRun:
    rng = np.random.default_rng(seed)
    center = np.full(dim, 0.5)
    best = None
    evaluations = 0
    batch = min(48, budget)
    scale = 0.30
    while evaluations < budget:
        n = min(batch, budget - evaluations)
        pop = np.clip(center + rng.normal(scale=scale, size=(n, dim)), 0.0, 1.0)
        value, unc, valid = objective(pop)
        score = np.where(valid, value, -np.inf)
        idx = int(np.argmax(score))
        evaluations += n
        if best is None or score[idx] > best[0]:
            best = (float(score[idx]), pop[idx].copy(), float(unc[idx]), bool(valid[idx]))
            center = pop[idx]
        scale = max(scale * 0.72, 0.02)
    assert best is not None
    return StrategyRun("linear_local", best[1], best[0], best[2], best[3], evaluations)


def _mappo_search(
    objective: Objective,
    dim: int,
    seed: int,
    budget: int,
    differentiable_step: Callable | None,
) -> StrategyRun:
    torch.manual_seed(seed)
    groups = dim
    obs_dim = 2
    adjacency = torch.eye(groups)
    policy = SharedGraphMAPPO(obs_dim=obs_dim, action_dim=1, adjacency=adjacency, hidden=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    episodes = max(4, min(20, budget // 8))
    batch = max(4, min(8, budget // episodes))
    evaluations = 0
    best: tuple[float, np.ndarray, float, bool] | None = None
    state = torch.zeros(batch, groups, obs_dim)
    for _ in range(episodes):
        action, logp, value = policy.act(state)
        if differentiable_step is not None:
            next_state, reward, valid_t = differentiable_step(state, action)
            reward = reward.float()
            valid_t = valid_t.bool()
        else:
            next_state = 0.8 * state + 0.2 * action.expand(-1, -1, obs_dim)
            reward = 1.0 - (action - 0.3).pow(2).mean(dim=(1, 2))
            valid_t = torch.ones(batch, dtype=torch.bool)
        advantage = reward - value.detach()
        returns = reward.detach()
        loss = policy.ppo_loss(state, action.detach(), logp.detach(), advantage.detach(), returns)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        state = next_state.detach()
        candidates = ((action.detach().cpu().numpy().squeeze(-1) + 1.0) / 2.0).clip(0.05, 0.95)
        values, unc, valid = objective(candidates)
        valid = np.asarray(valid, bool) & valid_t.cpu().numpy()
        for i in range(len(candidates)):
            if valid[i] and (best is None or values[i] > best[0]):
                best = (float(values[i]), candidates[i].copy(), float(unc[i]), True)
        evaluations += len(candidates)
        if evaluations >= budget:
            break
    if best is None:
        fallback = np.full(dim, 0.5)
        value, unc, valid = objective(fallback[None, :])
        best = (float(value[0]), fallback, float(unc[0]), bool(valid[0]))
        evaluations += 1
    return StrategyRun("graph_mappo", best[1], best[0], best[2], best[3], evaluations)


def run_blackbox_strategy(
    name: str,
    objective: Objective,
    *,
    dim: int,
    seed: int = 42,
    budget: int = 512,
    differentiable_step: Callable | None = None,
) -> StrategyRun:
    if name == "linear_local":
        return _local_search(objective, dim, seed, budget)
    iterations = max(2, min(10, int(np.sqrt(max(budget, 4)))))
    population = max(4, budget // iterations)
    if name == "gru_cem":
        r = risk_aware_cem(objective, dim=dim, seed=seed, population=population, iterations=iterations, risk_beta=0.0)
    elif name == "tcn_cma":
        r = diagonal_cma_es(objective, dim=dim, seed=seed, population=population, iterations=iterations, risk_beta=0.10)
    elif name == "graph_risk_cem":
        r = risk_aware_cem(objective, dim=dim, seed=seed, population=population, iterations=iterations, risk_beta=0.35)
    elif name == "graph_mappo":
        return _mappo_search(objective, dim, seed, budget, differentiable_step)
    else:
        raise KeyError(f"unknown strategy: {name}")
    value, unc, valid = objective(r.x[None, :])
    return StrategyRun(name, r.x, float(value[0]), float(unc[0]), bool(valid[0]), r.evaluations)
