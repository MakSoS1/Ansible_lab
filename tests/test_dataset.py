from pathlib import Path

from aios_track2.dataset import split_scenarios
from aios_track2.deck import build_well_graph, parse_deck
from aios_track2.doe import DoeConfig, generate_scenarios, schedule_hash


def test_scenario_split_has_no_overlap() -> None:
    split = split_scenarios([f"s{i:03d}" for i in range(20)], seed=42)
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert split == split_scenarios([f"s{i:03d}" for i in range(20)], seed=42)


def test_sobol_is_repeatable() -> None:
    graph = build_well_graph(parse_deck(Path("tests/fixtures/minimal.DATA")), radius_m=10_000)
    config = DoeConfig(n_scenarios=4, n_quarters=4, max_quarterly_change=0.2, cluster_count=2, seed=42)
    first = generate_scenarios(config, graph)
    second = generate_scenarios(config, graph)
    assert [schedule_hash(item) for item in first] == [schedule_hash(item) for item in second]
