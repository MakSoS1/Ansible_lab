"""Pin the three-way population split and the profile it reports.

The conclusion this audit feeds — that the human slice is not the population
we are scored on — is only as good as the membership test underneath it, so
the disjointness and the streaming accumulator are checked on a fixture whose
answers can be counted by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecup_matching.ml.run_v17_population_audit import run_population_audit


@pytest.fixture()
def fixture_root(tmp_path):
    items = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["aaaa", "bbbb", "cc", "dd", "e", "ffffff"],
            "attributes": [
                '{"brand":"x","model":"y"}',
                '{"brand":"z"}',
                "{}",
                "",
                '{"a":"1","b":"2","c":"3"}',
                "{}",
            ],
            "category": ["A", "A", "B", "B", "B", "A"],
        }
    )
    items.to_parquet(tmp_path / "items.parquet", index=False)
    pd.DataFrame({"id": [1, 2], "category": ["A", "A"]}).to_parquet(
        tmp_path / "items_human.parquet", index=False
    )
    # Weak pool references 3 and 4 only; 5 and 6 belong to neither population.
    pd.DataFrame({"id1": [3], "id2": [4]}).to_parquet(
        tmp_path / "matches_llm.parquet", index=False
    )
    return tmp_path


def _run(root):
    return run_population_audit(
        items_path=root / "items.parquet",
        human_items_path=root / "items_human.parquet",
        weak_matches_path=root / "matches_llm.parquet",
        output_path=root / "population.json",
    )


def test_populations_partition_the_universe_without_overlap(fixture_root):
    report = _run(fixture_root)
    sizes = {name: part["rows"] for name, part in report["populations"].items()}
    assert sizes == {"human": 2, "weak": 2, "unlabelled": 2}
    assert sum(sizes.values()) == report["items_rows"] == 6


def test_attribute_density_is_measured_per_population(fixture_root):
    populations = _run(fixture_root)["populations"]
    # ids 1,2 carry 3 JSON keys between them; ids 3,4 carry none.
    assert populations["human"]["attribute_keys_mean"] == pytest.approx(1.5)
    assert populations["weak"]["attribute_keys_mean"] == pytest.approx(0.0)
    assert populations["weak"]["attribute_missing_fraction"] == pytest.approx(1.0)
    assert populations["unlabelled"]["attribute_keys_mean"] == pytest.approx(1.5)


def test_name_length_and_category_share_are_reported(fixture_root):
    populations = _run(fixture_root)["populations"]
    assert populations["human"]["name_chars_mean"] == pytest.approx(4.0)
    assert populations["human"]["category_share"] == {"A": pytest.approx(1.0)}
    assert populations["weak"]["category_share"] == {"B": pytest.approx(1.0)}


def test_missing_weak_pool_puts_everything_outside_human_in_unlabelled(fixture_root):
    report = run_population_audit(
        items_path=fixture_root / "items.parquet",
        human_items_path=fixture_root / "items_human.parquet",
        weak_matches_path=None,
        output_path=fixture_root / "population-noweak.json",
    )
    sizes = {name: part["rows"] for name, part in report["populations"].items()}
    assert sizes == {"human": 2, "weak": 0, "unlabelled": 4}
