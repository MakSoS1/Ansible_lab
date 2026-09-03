from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Objective = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True, slots=True)
class SearchResult:
    x: np.ndarray
    score: float
    uncertainty: float
    evaluations: int
    history: tuple[float, ...]


def _ranked_score(value: np.ndarray, uncertainty: np.ndarray, valid: np.ndarray, beta: float) -> np.ndarray:
    score = np.asarray(value, float) - beta * np.asarray(uncertainty, float)
    return np.where(np.asarray(valid, bool), score, -np.inf)


def risk_aware_cem(
    objective: Objective,
    *,
    dim: int,
    seed: int = 42,
    population: int = 128,
    iterations: int = 20,
    elite_fraction: float = 0.15,
    risk_beta: float = 0.25,
    lower: float = 0.0,
    upper: float = 1.0,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    mean = np.full(dim, (lower + upper) / 2)
    std = np.full(dim, (upper - lower) / 3)
    elite_n = max(2, int(population * elite_fraction))
    best_x = mean.copy()
    best_score = -np.inf
    best_unc = np.inf
    history: list[float] = []
    for _ in range(iterations):
        pop = np.clip(rng.normal(mean, std, size=(population, dim)), lower, upper)
        value, unc, valid = objective(pop)
        score = _ranked_score(value, unc, valid, risk_beta)
        idx = np.argsort(score)[-elite_n:]
        finite = idx[np.isfinite(score[idx])]
        if len(finite):
            elite = pop[finite]
            mean = 0.35 * mean + 0.65 * elite.mean(axis=0)
            std = np.maximum(0.35 * std + 0.65 * elite.std(axis=0), (upper - lower) * 0.01)
            top = int(finite[np.argmax(score[finite])])
            if score[top] > best_score:
                best_x, best_score, best_unc = pop[top].copy(), float(score[top]), float(unc[top])
        history.append(float(best_score))
    return SearchResult(best_x, best_score, best_unc, population * iterations, tuple(history))


def diagonal_cma_es(
    objective: Objective,
    *,
    dim: int,
    seed: int = 42,
    population: int = 64,
    iterations: int = 30,
    risk_beta: float = 0.0,
    lower: float = 0.0,
    upper: float = 1.0,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    mean = np.full(dim, 0.5 * (lower + upper), dtype=float)
    sigma = 0.30 * (upper - lower)
    diag = np.ones(dim)
    mu = max(2, population // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    best_x = mean.copy()
    best_score = -np.inf
    best_unc = np.inf
    hist: list[float] = []
    for _ in range(iterations):
        z = rng.normal(size=(population, dim))
        pop = np.clip(mean + sigma * z * np.sqrt(diag), lower, upper)
        value, unc, valid = objective(pop)
        score = _ranked_score(value, unc, valid, risk_beta)
        order = np.argsort(score)[::-1]
        elite_idx = order[:mu]
        finite = elite_idx[np.isfinite(score[elite_idx])]
        if len(finite):
            use = finite[: min(mu, len(finite))]
            ww = weights[:len(use)]
            ww = ww / ww.sum()
            old = mean.copy()
            mean = np.sum(pop[use] * ww[:, None], axis=0)
            normalized = (pop[use] - old) / max(sigma, 1e-12)
            diag = np.clip(0.8 * diag + 0.2 * np.sum(ww[:, None] * normalized**2, axis=0), 1e-4, 4.0)
            progress = np.linalg.norm(mean - old) / (np.sqrt(dim) * max(sigma, 1e-12))
            sigma *= float(np.exp(0.18 * (progress - 0.25)))
            sigma = float(np.clip(sigma, 0.01 * (upper - lower), 0.5 * (upper - lower)))
            top = int(use[0])
            if score[top] > best_score:
                best_x, best_score, best_unc = pop[top].copy(), float(score[top]), float(unc[top])
        hist.append(float(best_score))
    return SearchResult(best_x, best_score, best_unc, population * iterations, tuple(hist))
