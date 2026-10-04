from pathlib import Path

from aios_track2.deck import build_well_graph, parse_deck


def test_parse_dimensions_and_wells() -> None:
    metadata = parse_deck(Path("tests/fixtures/minimal.DATA"))
    assert metadata.dimensions == (3, 2, 1)
    assert [well.name for well in metadata.wells] == ["P1", "I1"]


def test_graph_is_symmetric() -> None:
    graph = build_well_graph(parse_deck(Path("tests/fixtures/minimal.DATA")), radius_m=10_000)
    assert set(graph.edges) == {("P1", "I1"), ("I1", "P1")}
