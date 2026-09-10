from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from aios_track2.economics import calculate_npv
from aios_track2.opm import FlowRequest, run_flow
from aios_track2.physics import ProxyFlow
from aios_track2.schedule import Control, Schedule, WellRole, project_schedule


def risk_score(mean_npv: float, std_npv: float, penalty: float = 1.0) -> float:
    return mean_npv - penalty * std_npv


@dataclass(frozen=True)
class Candidate:
    schedule: Schedule
    mean_npv: float
    std_npv: float
    accepted: bool
    opm_validated: bool
    schedule_sha256: str
    vector: tuple[float, ...] = ()
    ood: bool = False
    backend: str = "surrogate"


@dataclass
class OptimizationRequest:
    seed: int
    population: int
    iterations: int
    elites: int = 4
    noise: float = 0.15
    uncertainty_penalty: float = 1.0
    wells: tuple[str, ...] = ("P1", "I1")
    dates: tuple[date, ...] = (date(2007, 1, 1), date(2007, 4, 1))
    method: str = "cem"
    output_dir: Path | None = None


@dataclass
class OptimizationResult:
    best: Candidate
    candidates: tuple[Candidate, ...]
    method: str


class ToySurrogate:
    def __init__(self, target: float = 0.6) -> None:
        self.target = target

    def score(self, vector: np.ndarray) -> tuple[float, float]:
        error = float(((vector - self.target) ** 2).mean())
        return 100.0 - 400.0 * error, 5.0 * error


def _decode(vector: np.ndarray, wells: tuple[str, ...], dates: tuple[date, ...]) -> Schedule:
    controls: list[Control] = []
    n_wells = max(len(wells), 1)
    for time_index, day in enumerate(dates):
        for well_index, well in enumerate(wells):
            idx = (time_index * n_wells + well_index) % len(vector)
            rate = float(np.clip(vector[idx] * 200.0, 10.0, 480.0))
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


def _evaluate_vector(vector: np.ndarray, request: OptimizationRequest, surrogate: ToySurrogate | None) -> Candidate:
    schedule = _decode(vector, request.wells, request.dates)
    projection = project_schedule(schedule)
    if not projection.accepted:
        return Candidate(
            schedule=schedule,
            mean_npv=float("-inf"),
            std_npv=0.0,
            accepted=False,
            opm_validated=False,
            schedule_sha256=schedule.sha256,
            vector=tuple(float(x) for x in vector),
        )
    if surrogate is not None:
        mean, std = surrogate.score(vector)
        return Candidate(
            schedule=projection.projected,
            mean_npv=risk_score(mean, std, request.uncertainty_penalty),
            std_npv=std,
            accepted=True,
            opm_validated=False,
            schedule_sha256=projection.projected.sha256,
            vector=tuple(float(x) for x in vector),
        )
    monthly = ProxyFlow(seed=request.seed).run(projection.projected)
    npv = float(calculate_npv(monthly).npv_mrub)
    return Candidate(
        schedule=projection.projected,
        mean_npv=npv,
        std_npv=0.0,
        accepted=True,
        opm_validated=False,
        schedule_sha256=projection.projected.sha256,
        vector=tuple(float(x) for x in vector),
        backend="proxy",
    )


def _iterate(request: OptimizationRequest, method: str, surrogate: ToySurrogate | None) -> OptimizationResult:
    rng = np.random.default_rng(request.seed + (7 if method == "cma" else 0))
    dim = max(len(request.wells) * len(request.dates), 4)
    mean = np.full(dim, 0.55)
    std = np.full(dim, request.noise)
    cov = np.eye(dim)
    sigma = 0.2
    hall: list[Candidate] = []
    for _ in range(request.iterations):
        if method == "cma":
            population = np.clip(rng.multivariate_normal(mean, (sigma**2) * cov, size=request.population), 0.05, 0.95)
        else:
            population = np.clip(rng.normal(mean, std, size=(request.population, dim)), 0.05, 0.95)
        scored = [_evaluate_vector(vector, request, surrogate) for vector in population]
        pairs = [(vector, item) for vector, item in zip(population, scored, strict=True) if item.accepted]
        pairs.sort(key=lambda pair: pair[1].mean_npv, reverse=True)
        hall.extend(item for _, item in pairs)
        elite = pairs[: max(1, request.elites)]
        if elite:
            stacked = np.stack([vector for vector, _ in elite])
            mean = stacked.mean(axis=0)
            std = np.maximum(stacked.std(axis=0), 0.02)
            centered = stacked - mean
            cov = (centered.T @ centered) / max(len(stacked), 1) + 1e-3 * np.eye(dim)
            sigma = max(0.05, 0.85 * sigma + 0.15 * float(np.mean(np.linalg.norm(centered, axis=1))))
    hall.sort(key=lambda item: item.mean_npv, reverse=True)
    unique: dict[str, Candidate] = {item.schedule_sha256: item for item in hall}
    ranked = tuple(sorted(unique.values(), key=lambda item: item.mean_npv, reverse=True))
    promoted = promote_candidates(ranked[:8], budget=1, seed=request.seed, output_dir=request.output_dir)
    champion = ranked[0]
    validated = promoted[0]
    best = Candidate(
        schedule=validated.schedule,
        mean_npv=champion.mean_npv if surrogate is not None else validated.mean_npv,
        std_npv=champion.std_npv,
        accepted=validated.accepted,
        opm_validated=validated.opm_validated,
        schedule_sha256=validated.schedule_sha256,
        vector=champion.vector,
        backend=validated.backend,
    )
    return OptimizationResult(best=best, candidates=ranked[:12], method=method)


def optimize(request: OptimizationRequest, surrogate: ToySurrogate | None = None) -> OptimizationResult:
    return _iterate(request, request.method, surrogate)


def promote_candidates(
    candidates: tuple[Candidate, ...] | list[Candidate],
    budget: int,
    seed: int,
    output_dir: Path | None = None,
) -> tuple[Candidate, ...]:
    root = Path(output_dir or tempfile.mkdtemp(prefix="aios-promote-"))
    promoted: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.schedule_sha256 in seen:
            continue
        seen.add(candidate.schedule_sha256)
        if len(promoted) >= budget:
            break
        work = root / candidate.schedule_sha256[:12]
        work.mkdir(parents=True, exist_ok=True)
        result = run_flow(
            FlowRequest(
                deck=work / "CASE.DATA",
                output_dir=work,
                schedule=candidate.schedule,
                seed=seed,
            )
        )
        npv = candidate.mean_npv
        if result.monthly_path is not None:
            import pandas as pd

            npv = float(calculate_npv(pd.read_parquet(result.monthly_path)).npv_mrub)
        promoted.append(
            Candidate(
                schedule=candidate.schedule,
                mean_npv=npv,
                std_npv=candidate.std_npv,
                accepted=candidate.accepted and result.status == "success",
                opm_validated=result.status == "success",
                schedule_sha256=candidate.schedule_sha256,
                vector=candidate.vector,
                backend=result.backend,
            )
        )
    return tuple(promoted)


def active_learning_batch(candidates: tuple[Candidate, ...], exploit: int, explore: int) -> tuple[Candidate, ...]:
    ranked = sorted(candidates, key=lambda item: item.mean_npv, reverse=True)
    uncertain = sorted(candidates, key=lambda item: item.std_npv, reverse=True)
    selected: list[Candidate] = []
    for item in ranked[:exploit] + uncertain:
        if item not in selected and len(selected) < exploit + explore:
            selected.append(item)
    return tuple(selected)


class ToyOptimizer:
    def __init__(self, seed: int = 42) -> None:
        self.request = OptimizationRequest(seed=seed, population=16, iterations=6, elites=4, method="cem")

    def run(self) -> OptimizationResult:
        return optimize(self.request, surrogate=ToySurrogate())
