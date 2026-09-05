from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategySpec:
    name: str
    surrogate: str
    optimizer: str
    uncertainty_aware: bool
    active_learning: bool
    purpose: str


TOP5_STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec("linear_local", "linear", "local", False, False, "deterministic lower bound"),
    StrategySpec("gru_cem", "gru", "cem", False, False, "cheap temporal baseline"),
    StrategySpec("tcn_cma", "tcn", "diagonal-cma-es", True, False, "parallel temporal challenger"),
    StrategySpec("graph_risk_cem", "graph-temporal-ensemble", "risk-aware-cem", True, True, "primary competition strategy"),
    StrategySpec("graph_mappo", "graph-temporal-ensemble", "shared-graph-mappo-ctde", True, True, "MARL challenger"),
)


def strategy_names() -> tuple[str, ...]:
    return tuple(s.name for s in TOP5_STRATEGIES)
