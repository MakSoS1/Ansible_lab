from pathlib import Path

from aios_track2.deck import build_well_graph, parse_deck
from aios_track2.doe import DoeConfig, generate_scenarios
from aios_track2.schedule import project_schedule


def test_generated_schedules_are_feasible() -> None:
    graph = build_well_graph(parse_deck(Path("tests/fixtures/minimal.DATA")), radius_m=10_000)
    schedules = generate_scenarios(
        DoeConfig(n_scenarios=4, n_quarters=4, max_quarterly_change=0.2, cluster_count=2, seed=42),
        graph,
    )
    assert len(schedules) == 4
    for schedule in schedules:
        assert project_schedule(schedule).accepted
