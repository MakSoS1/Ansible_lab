from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import qmc

FROZEN_DESIGN_SHA256 = "571b00af32773c13df8dd4a9497f8096ad6588fd34dcc5fb73619fc042cb6b9a"
_DESIGN_SEED = 314159
_SPLIT_SEED = 271828
_LOWER = 0.8
_UPPER = 1.2


@dataclass(frozen=True, slots=True)
class RealOpmScenario:
    scenario_id: int
    split: str
    producer_2007: float
    producer_2025: float
    injector_2007: float
    injector_2025: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def frozen_real_doe() -> tuple[RealOpmScenario, ...]:
    """Return the target-independent 32-scenario Model Z validation design.

    The Sobol points, split permutation and bounds are deliberately constants.
    Scenario zero is the untouched baseline and is permanently assigned to the
    training set.  Holdout membership is therefore fixed before any OPM target
    is generated and cannot be selected after observing validation quality.
    """
    sampler = qmc.Sobol(d=4, scramble=True, seed=_DESIGN_SEED)
    points = _LOWER + (_UPPER - _LOWER) * sampler.random_base2(m=5)
    points[0] = 1.0

    rng = np.random.default_rng(_SPLIT_SEED)
    permutation = rng.permutation(np.arange(1, 32)).tolist()
    train = {0, *permutation[:19]}
    validation = set(permutation[19:23])

    scenarios: list[RealOpmScenario] = []
    for scenario_id, row in enumerate(points):
        split = "train" if scenario_id in train else "validation" if scenario_id in validation else "holdout"
        values = tuple(round(float(value), 12) for value in row)
        scenarios.append(RealOpmScenario(scenario_id, split, *values))
    return tuple(scenarios)


def scenario_by_id(scenario_id: int) -> RealOpmScenario:
    scenarios = frozen_real_doe()
    if not 0 <= scenario_id < len(scenarios):
        raise ValueError(f"scenario_id must be in [0, {len(scenarios) - 1}]")
    return scenarios[scenario_id]
