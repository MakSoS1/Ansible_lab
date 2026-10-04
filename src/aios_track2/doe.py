from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

from aios_track2.deck import WellGraph
from aios_track2.schedule import ConstraintSet, Control, Schedule, WellRole, project_schedule


@dataclass(frozen=True)
class DoeConfig:
    n_scenarios: int
    n_quarters: int
    max_quarterly_change: float
    cluster_count: int
    seed: int

    @classmethod
    def from_yaml(cls, path: Path) -> DoeConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**payload)


def _quarter_starts(n_quarters: int, start: date = date(2007, 1, 1)) -> list[date]:
    dates = []
    year, month = start.year, start.month
    for _ in range(n_quarters):
        dates.append(date(year, month, 1))
        month += 3
        if month > 12:
            year += 1
            month -= 12
    return dates


def generate_scenarios(
    config: DoeConfig,
    graph: WellGraph,
    constraints: ConstraintSet | None = None,
    base_liquid: float = 80.0,
    base_injection: float = 90.0,
) -> tuple[Schedule, ...]:
    constraints = constraints or ConstraintSet(known_wells=frozenset(well.name for well in graph.wells))
    clusters = graph.clusters or (tuple(well.name for well in graph.wells),)
    n_clusters = len(clusters)
    dim = n_clusters * 2
    sampler = qmc.Sobol(d=dim, scramble=True, seed=config.seed)
    needed = config.n_scenarios
    raw = sampler.random_base2(m=max(1, int(np.ceil(np.log2(max(needed, 2))))))
    schedules: list[Schedule] = []
    knots = _quarter_starts(config.n_quarters)
    index = 0
    attempts = 0
    while len(schedules) < needed and attempts < needed * 8:
        vector = raw[index % len(raw)]
        if attempts:
            vector = (vector + 0.07 * attempts) % 1.0
        index += 1
        attempts += 1
        controls: list[Control] = []
        factors = np.ones(n_clusters)
        inj_factors = np.ones(n_clusters)
        for knot_i, knot in enumerate(knots):
            for cluster_i, names in enumerate(clusters):
                offset = cluster_i * 2
                delta = (vector[offset] - 0.5) * 2 * config.max_quarterly_change
                inj_delta = (vector[offset + 1] - 0.5) * 2 * config.max_quarterly_change
                if knot_i:
                    factors[cluster_i] = np.clip(factors[cluster_i] * (1.0 + delta), 0.4, 1.6)
                    inj_factors[cluster_i] = np.clip(inj_factors[cluster_i] * (1.0 + inj_delta), 0.4, 1.8)
                for name in names:
                    well = next(item for item in graph.wells if item.name == name)
                    is_injector = well.phase.upper() == "WATER" or name.endswith("I") or well.phase.upper() == "INJ"
                    if is_injector:
                        controls.append(
                            Control(
                                date=knot,
                                well=name,
                                status="OPEN",
                                role=WellRole.INJECTOR,
                                wwir=float(base_injection * inj_factors[cluster_i]),
                            )
                        )
                    else:
                        controls.append(
                            Control(
                                date=knot,
                                well=name,
                                status="OPEN",
                                role=WellRole.PRODUCER,
                                wlpr=float(base_liquid * factors[cluster_i]),
                            )
                        )
        candidate = Schedule(controls=tuple(controls))
        result = project_schedule(candidate, constraints)
        if result.accepted:
            schedules.append(result.projected)
    if len(schedules) < needed:
        raise RuntimeError("Sobol sampler could not fill the requested feasible set")
    return tuple(schedules[:needed])


def schedule_hash(schedule: Schedule) -> str:
    from aios_track2.schedule import write_schedule_text

    return sha256(write_schedule_text(schedule).encode("utf-8")).hexdigest()
