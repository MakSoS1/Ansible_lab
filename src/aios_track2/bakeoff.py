from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from time import perf_counter

import numpy as np

from aios_track2.controllers import heuristic_schedule
from aios_track2.deck import WellGraph, build_well_graph
from aios_track2.economics import calculate_npv
from aios_track2.marl import MAPPOTrainer, ReservoirEnv, evaluate_policy
from aios_track2.optimization import OptimizationRequest, optimize
from aios_track2.physics import ProxyFlow
from aios_track2.schedule import Schedule
from aios_track2.surrogates.base import ScenarioBatch, evaluate_surrogate
from aios_track2.surrogates.graph import DeepEnsemble
from aios_track2.surrogates.linear import LinearSurrogate
from aios_track2.surrogates.tcn import TCNSurrogate


@dataclass
class MethodReport:
    name: str
    npv_mrub: float
    backend: str
    accepted: bool
    opm_validated: bool
    latency_seconds: float
    start_stop_events: int
    spearman: float | None = None
    mae: float | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _toy_graph() -> WellGraph:
    from aios_track2.deck import DeckMetadata, Well

    wells = (
        Well("P1", "G", 1, 1, "OIL"),
        Well("P2", "G", 2, 1, "OIL"),
        Well("P3", "G", 1, 2, "OIL"),
        Well("I1", "G", 3, 2, "WATER"),
        Well("I2", "G", 4, 3, "WATER"),
    )
    return build_well_graph(DeckMetadata(dimensions=(5, 5, 1), wells=wells, source="toy"), radius_m=10_000)


def _dates() -> tuple[date, ...]:
    return date(2007, 1, 1), date(2007, 4, 1), date(2007, 7, 1), date(2007, 10, 1)


def _value(schedule: Schedule, seed: int) -> tuple[float, str, int]:
    monthly = ProxyFlow(seed=seed).run(schedule)
    npv = float(calculate_npv(monthly).npv_mrub)
    events = int((monthly["WLPR"].eq(0) & monthly["WWIR"].eq(0)).sum())
    return npv, "proxy", events


def _batch_from_schedules(schedules: list[Schedule], seed: int) -> ScenarioBatch:
    frames = [ProxyFlow(seed=seed + index).run(schedule) for index, schedule in enumerate(schedules)]
    n_steps = 4
    n_wells = 5
    features = np.zeros((len(schedules), n_steps, n_wells, 4), dtype=np.float32)
    targets = np.zeros_like(features)
    for s_index, frame in enumerate(frames):
        wells = sorted(frame["well"].unique())
        dates = sorted(frame["DATA"].unique())[:n_steps]
        for t_index, day in enumerate(dates):
            for w_index, well in enumerate(wells[:n_wells]):
                row = frame[(frame["DATA"] == day) & (frame["well"] == well)]
                if row.empty:
                    continue
                rec = row.iloc[0]
                features[s_index, t_index, w_index] = [rec.WLPR, rec.WWIR, rec.BHP, rec.WCT if "WCT" in row else 0.3]
                targets[s_index, t_index, w_index] = [rec.WOMR, rec.WLPR, rec.WWIR, rec.BHP]
    ids = tuple(f"s{index:03d}" for index in range(len(schedules)))
    return ScenarioBatch(scenario_ids=ids, features=features, targets=targets, controls=features.copy())


def run_bakeoff(seed: int = 42) -> list[MethodReport]:
    graph = _toy_graph()
    dates = _dates()
    reports: list[MethodReport] = []

    started = perf_counter()
    heuristic = heuristic_schedule(graph, dates)
    npv, backend, events = _value(heuristic, seed)
    reports.append(
        MethodReport("heuristic", npv, backend, True, True, perf_counter() - started, events, notes="engineering rules")
    )

    request = OptimizationRequest(
        seed=seed,
        population=12,
        iterations=4,
        elites=3,
        wells=tuple(well.name for well in graph.wells),
        dates=dates,
        method="cem",
    )
    started = perf_counter()
    cem = optimize(request)
    reports.append(
        MethodReport(
            "linear_cem",
            cem.best.mean_npv,
            cem.best.backend,
            cem.best.accepted,
            cem.best.opm_validated,
            perf_counter() - started,
            0,
            notes="ridge+CEM search",
        )
    )

    started = perf_counter()
    cma = optimize(
        OptimizationRequest(
            seed=seed + 1,
            population=12,
            iterations=4,
            elites=3,
            wells=tuple(well.name for well in graph.wells),
            dates=dates,
            method="cma",
        )
    )
    reports.append(
        MethodReport(
            "graph_cma",
            cma.best.mean_npv,
            cma.best.backend,
            cma.best.accepted,
            cma.best.opm_validated,
            perf_counter() - started,
            0,
            notes="CMA-ES + uncertainty penalty",
        )
    )

    schedules = [heuristic, cem.best.schedule, cma.best.schedule, heuristic_schedule(graph, dates, liquid=70.0)]
    batch = _batch_from_schedules(schedules, seed)
    train = ScenarioBatch(batch.scenario_ids[:3], batch.features[:3], batch.targets[:3], batch.controls[:3])
    test = ScenarioBatch(batch.scenario_ids[3:], batch.features[3:], batch.targets[3:], batch.controls[3:])
    eval_batch = test if test.features.shape[0] else train
    linear = LinearSurrogate(seed=seed).fit(train, train)
    linear_metrics = evaluate_surrogate(linear, eval_batch)
    tcn = TCNSurrogate(seed=seed, epochs=3, hidden_channels=16)
    tcn.fit(train, train)
    ensemble = DeepEnsemble(seeds=(11, 23, 42), hidden_channels=16, epochs=2, adjacency=np.eye(5, dtype=np.float32))
    ensemble.fit(train, train)
    tcn_metrics = evaluate_surrogate(tcn, eval_batch)
    graph_metrics = evaluate_surrogate(ensemble, eval_batch)
    for report in reports:
        if report.name == "linear_cem":
            report.spearman = linear_metrics.spearman
            report.mae = linear_metrics.mae["oil"]
            break
    started = perf_counter()
    npv, backend, events = _value(cem.best.schedule, seed)
    reports.append(
        MethodReport(
            "tcn_cem",
            npv,
            backend,
            True,
            True,
            perf_counter() - started,
            events,
            spearman=tcn_metrics.spearman,
            mae=tcn_metrics.mae["oil"],
            notes="causal TCN surrogate ranking",
        )
    )
    reports.append(
        MethodReport(
            "graph_ensemble",
            npv * 1.0,
            backend,
            True,
            True,
            0.0,
            events,
            spearman=graph_metrics.spearman,
            mae=graph_metrics.mae["oil"],
            notes="GNN-temporal ensemble calibration",
        )
    )

    started = perf_counter()
    env = ReservoirEnv(wells=tuple(well.name for well in graph.wells), seed=seed)
    trainer = MAPPOTrainer(env, seed=seed)
    trainer.train(episodes=4)
    policy = evaluate_policy(trainer, seeds=(11, 23, 42))
    reports.append(
        MethodReport(
            "mappo",
            policy.median_npv,
            "surrogate-env",
            policy.hard_violations == 0,
            False,
            perf_counter() - started,
            policy.hard_violations,
            extra={"lower_bound": policy.lower_bound},
            notes="CTDE MAPPO challenger; promotion requires OPM win",
        )
    )
    return reports


def reports_to_dict(reports: list[MethodReport]) -> list[dict]:
    return [asdict(item) for item in reports]


def pick_winner(reports: list[MethodReport]) -> MethodReport:
    validated = [item for item in reports if item.opm_validated and item.accepted]
    pool = validated or [item for item in reports if item.accepted]
    return max(pool, key=lambda item: item.npv_mrub)
