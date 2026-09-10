from __future__ import annotations

import json

import numpy as np

from aios_track2.strategies import strategy_names
from aios_track2.strategy_runner import run_blackbox_strategy


def objective(pop: np.ndarray):
    target = np.linspace(0.35, 0.75, pop.shape[1])
    value = 12.0 - np.mean((pop - target) ** 2, axis=1)
    uncertainty = 0.02 + 0.12 * np.mean(np.abs(pop - 0.5), axis=1)
    valid = np.all((pop >= 0.04) & (pop <= 0.96), axis=1)
    return value, uncertainty, valid


def main() -> None:
    rows = []
    for idx, name in enumerate(strategy_names()):
        run = run_blackbox_strategy(name, objective, dim=12, seed=42 + idx, budget=256)
        rows.append({
            "strategy": run.strategy,
            "score": run.score,
            "uncertainty": run.uncertainty,
            "valid": run.valid,
            "evaluations": run.evaluations,
        })
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
