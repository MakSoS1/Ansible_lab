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
    """Optimize the real black-box objective with a shared PPO actor.

    When a differentiable environment is supplied its reward remains the PPO
    training signal. Otherwise the policy is trained directly from the same
    risk-adjusted objective used to rank competition candidates. This avoids
    the previous synthetic action≈0.3 demo reward, which was unrelated to
    reservoir economics.
    """
    torch.manual_seed(seed)
    groups = dim
    obs_dim = 2
    adjacency = torch.eye(groups)
    policy = SharedGraphMAPPO(obs_dim=obs_dim, action_dim=1, adjacency=adjacency, hidden=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    episodes = max(4, min(64, max(1, budget // 16)))
    batch = max(4, min(64, max(1, budget // episodes)))
    evaluations = 0
    best_rank = -np.inf
    best: tuple[float, np.ndarray, float, bool] | None = None
    state = torch.zeros(batch, groups, obs_dim)
    risk_beta = 0.25

    for _ in range(episodes):
        if evaluations >= budget:
            break
        action, logp, value_estimate = policy.act(state)
        candidates = ((action.detach().cpu().numpy().squeeze(-1) + 1.0) / 2.0).clip(0.0, 1.0)
        remaining = budget - evaluations
        if len(candidates) > remaining:
            candidates = candidates[:remaining]
            action = action[:remaining]
            logp = logp[:remaining]
            value_estimate = value_estimate[:remaining]
            state = state[:remaining]

        values, unc, valid = objective(candidates)
        values = np.asarray(values, dtype=float).reshape(-1)
        unc = np.asarray(unc, dtype=float).reshape(-1)
        valid = np.asarray(valid, dtype=bool).reshape(-1)
        risk_score = values - risk_beta * unc

        if differentiable_step is not None:
            next_state, policy_reward, valid_t = differentiable_step(state, action)
            valid = valid & valid_t.detach().cpu().numpy().astype(bool)
            reward = policy_reward.float().reshape(-1)
        else:
            finite = valid & np.isfinite(risk_score)
            reward_np = np.full(len(candidates), -5.0, dtype=np.float32)
            if np.any(finite):
                center = float(np.mean(risk_score[finite]))
                scale = max(float(np.std(risk_score[finite])), 1e-6)
                reward_np[finite] = ((risk_score[finite] - center) / scale).astype(np.float32)
            reward = torch.as_tensor(reward_np, dtype=torch.float32, device=state.device)
            next_state = 0.8 * state + 0.2 * action.expand(-1, -1, obs_dim)

        advantage = reward - value_estimate.detach()
        returns = reward.detach()
        loss = policy.ppo_loss(state, action.detach(), logp.detach(), advantage.detach(), returns)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

        for index in range(len(candidates)):
            if valid[index] and np.isfinite(risk_score[index]) and risk_score[index] > best_rank:
                best_rank = float(risk_score[index])
                best = (float(values[index]), candidates[index].copy(), float(unc[index]), True)
        evaluations += len(candidates)
        state = next_state.detach()

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
