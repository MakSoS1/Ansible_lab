from __future__ import annotations

from datetime import date

from .eclipse_schedule import _interpolate_monthly, _scale_schedule


def scale_schedule_with_role_policies(
    text: str,
    *,
    producer_well_groups: dict[str, int],
    injector_well_groups: dict[str, int],
    producer_group_nodes: dict[int, tuple[float, ...]],
    injector_group_nodes: dict[int, tuple[float, ...]],
    node_dates: tuple[date, ...],
    effective_from: date,
    max_wlpr: float = 500.0,
) -> str:
    if not node_dates:
        raise ValueError("at least one policy node is required")
    for mapping in (producer_group_nodes, injector_group_nodes):
        for values in mapping.values():
            if len(values) != len(node_dates):
                raise ValueError("every group policy must provide one value per node date")
            if any(value <= 0 for value in values):
                raise ValueError("policy scales must be positive")
    missing_producer = sorted(set(producer_well_groups.values()) - set(producer_group_nodes))
    missing_injector = sorted(set(injector_well_groups.values()) - set(injector_group_nodes))
    if missing_producer:
        raise ValueError(f"missing producer group policies: {missing_producer}")
    if missing_injector:
        raise ValueError(f"missing injector group policies: {missing_injector}")

    def provider(current_date: date | None) -> tuple[dict[str, float], dict[str, float]]:
        if current_date is None:
            return {}, {}
        producer = {
            well: _interpolate_monthly(current_date, node_dates, producer_group_nodes[group])
            for well, group in producer_well_groups.items()
        }
        injector = {
            well: _interpolate_monthly(current_date, node_dates, injector_group_nodes[group])
            for well, group in injector_well_groups.items()
        }
        return producer, injector

    return _scale_schedule(
        text,
        scale_provider=provider,
        max_wlpr=max_wlpr,
        effective_from=effective_from,
    )
