from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from aios_track2.schedule import ConstraintSet, Control, Schedule, WellRole, project_schedule


class SpySurrogate:
    def __init__(self) -> None:
        self.last_batch: SimpleNamespace | None = None

    def predict_npv(self, schedule: Schedule) -> tuple[float, float]:
        max_wlpr = max((control.wlpr for control in schedule.controls), default=0.0)
        self.last_batch = SimpleNamespace(controls=SimpleNamespace(max_wlpr=max_wlpr), schedule=schedule)
        rates = [control.wlpr for control in schedule.controls]
        return 50.0 - 0.01 * float(np.var(rates) if rates else 0.0), 1.0


class ReservoirEnv:
    def __init__(
        self,
        wells: tuple[str, ...] = ("P1", "P2", "I1"),
        n_clusters: int = 2,
        horizon: int = 8,
        seed: int = 42,
        surrogate: SpySurrogate | None = None,
    ) -> None:
        self.wells = wells
        self.n_clusters = n_clusters
        self.horizon = horizon
        self.seed = seed
        self.surrogate = surrogate or SpySurrogate()
        self.constraints = ConstraintSet(known_wells=frozenset(wells), max_wlpr_m3_day=500.0)
        self._t = 0
        self._npv = 0.0

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.default_rng(seed)
        self._t = 0
        self._npv = 0.0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return np.array([self._t / self.horizon, self._npv / 100.0], dtype=np.float32)

    def safe_action(self) -> np.ndarray:
        return np.full(len(self.wells), 0.4, dtype=np.float32)

    def action_with_wlpr(self, wlpr: float) -> np.ndarray:
        return np.full(len(self.wells), wlpr / 200.0, dtype=np.float32)

    def _decode(self, action: np.ndarray) -> Schedule:
        day = date(2007, 1 + min(self._t * 3, 9), 1)
        controls = []
        for well, value in zip(self.wells, np.asarray(action, dtype=float), strict=False):
            rate = float(value * 200.0)
            injector = well.startswith("I")
            controls.append(
                Control(
                    date=day,
                    well=well,
                    status="OPEN",
                    role=WellRole.INJECTOR if injector else WellRole.PRODUCER,
                    wlpr=0.0 if injector else rate,
                    wwir=rate if injector else 0.0,
                )
            )
        return Schedule(controls=tuple(controls))

    def step(self, action: np.ndarray):
        projected = project_schedule(self._decode(np.asarray(action, dtype=float)), self.constraints)
        constraint_cost = 0.0 if projected.accepted else 25.0
        mean, std = self.surrogate.predict_npv(projected.projected)
        uncertainty_cost = 0.5 * std
        npv_delta = mean / self.horizon
        self._npv += npv_delta
        self._t += 1
        reward = npv_delta - constraint_cost - uncertainty_cost
        terminated = self._t >= self.horizon
        info = {
            "npv_delta": npv_delta,
            "constraint_cost": constraint_cost,
            "uncertainty_cost": uncertainty_cost,
        }
        return self._obs(), reward, terminated, False, info


class ClusterActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, act_dim), nn.Sigmoid())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralCritic(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MAPPOTrainer:
    def __init__(self, env: ReservoirEnv, seed: int = 42) -> None:
        torch.manual_seed(seed)
        self.env = env
        self.actor = ClusterActor(2, len(env.wells))
        self.critic = CentralCritic(2)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def train(self, episodes: int = 8) -> float:
        last = 0.0
        for _ in range(episodes):
            obs, _ = self.env.reset()
            rewards: list[float] = []
            logps: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            done = False
            while not done:
                tensor = torch.tensor(obs, dtype=torch.float32)
                dist_mean = self.actor(tensor)
                action = dist_mean.detach().numpy()
                value = self.critic(tensor)
                obs, reward, done, _, _ = self.env.step(action)
                rewards.append(reward)
                logps.append(torch.log(dist_mean.clamp(1e-4, 1.0)).mean())
                values.append(value)
            acc = 0.0
            returns = []
            for reward in reversed(rewards):
                acc = reward + 0.99 * acc
                returns.append(acc)
            returns_t = torch.tensor(list(reversed(returns)), dtype=torch.float32)
            val = torch.stack(values).squeeze()
            adv = returns_t - val.detach()
            loss_a = -(torch.stack(logps) * adv).mean()
            loss_c = torch.nn.functional.mse_loss(val, returns_t)
            self.opt_actor.zero_grad()
            loss_a.backward()
            self.opt_actor.step()
            self.opt_critic.zero_grad()
            loss_c.backward()
            self.opt_critic.step()
            last = float(sum(rewards))
        return last


@dataclass(frozen=True)
class PolicyReport:
    median_npv: float
    lower_bound: float
    hard_violations: int
    seeds: tuple[int, ...]


def evaluate_policy(policy: MAPPOTrainer, seeds: tuple[int, ...] = (11, 23, 42, 71, 101)) -> PolicyReport:
    scores = []
    violations = 0
    for seed in seeds:
        obs, _ = policy.env.reset(seed=seed)
        total = 0.0
        done = False
        while not done:
            with torch.no_grad():
                action = policy.actor(torch.tensor(obs, dtype=torch.float32)).numpy()
            obs, reward, done, _, info = policy.env.step(action)
            total += info["npv_delta"]
            if info["constraint_cost"] > 0:
                violations += 1
        scores.append(total)
    array = np.array(scores)
    return PolicyReport(
        median_npv=float(np.median(array)),
        lower_bound=float(np.percentile(array, 10)),
        hard_violations=violations,
        seeds=seeds,
    )


def marl_is_finalist(report: PolicyReport, best_cem_npv: float) -> bool:
    return report.median_npv > best_cem_npv and report.lower_bound > 0 and report.hard_violations == 0
