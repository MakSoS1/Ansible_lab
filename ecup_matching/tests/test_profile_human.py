import json
from pathlib import Path

import pandas as pd
import pytest

from ecup_matching.profile_human import build_profile, render_markdown


def _write_fixture(tmp_path: Path):
    items = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": [
                "Phone X 128GB",
                "phone x 128 gb",
                "Case red",
                "Case blue",
            ],
            "attributes": [
                '{"memory":"128 GB","color":"black"}',
                '{"memory":"128GB","color":"black"}',
                '{"color":"red"}',
                '{"color":"blue"}',
            ],
            "category": ["phones", "phones", "cases", "cases"],
        }
    )
    matches = pd.DataFrame(
        {
            "id1": [1, 1, 3],
            "id2": [2, 2, 4],
            "target": [1, 1, 0],
        }
    )
    items_path = tmp_path / "items_human.parquet"
    matches_path = tmp_path / "matches.parquet"
    items.to_parquet(items_path, index=False)
    matches.to_parquet(matches_path, index=False)
    return matches_path, items_path


def test_build_profile_reports_aggregate_pair_and_item_statistics(tmp_path):
    matches_path, items_path = _write_fixture(tmp_path)

    profile = build_profile(matches_path, items_path)

    assert profile["matches"]["rows"] == 3
    assert profile["matches"]["positive_rate"] == pytest.approx(2 / 3)
    assert profile["matches"]["target_counts"] == {"0": 1, "1": 2}
    assert profile["items"]["rows"] == 4
    assert profile["items"]["categories"] == 2
    assert profile["pair_categories"]["same_category_rate"] == pytest.approx(1.0)
    assert set(profile["category_breakdown"]) == {"cases", "phones"}
    assert profile["category_breakdown"]["phones"]["pairs"] == 2
    assert profile["category_breakdown"]["phones"]["positive_rate"] == pytest.approx(1.0)
    assert profile["items"]["distinct_normalized_names"] == 3
    assert profile["items"]["duplicate_normalized_name_rate"] == pytest.approx(0.25)
    assert profile["warnings"] == []


def test_profile_warns_about_missing_items_and_cross_category_pairs(tmp_path):
    items = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["a", "b"],
            "attributes": ["{}", "{}"],
            "category": ["cat-a", "cat-b"],
        }
    )
    matches = pd.DataFrame(
        {
            "id1": [1, 1],
            "id2": [2, 99],
            "target": [0, 0],
        }
    )
    items_path = tmp_path / "items.parquet"
    matches_path = tmp_path / "matches.parquet"
    items.to_parquet(items_path, index=False)
    matches.to_parquet(matches_path, index=False)

    profile = build_profile(matches_path, items_path)

    joined_warnings = " ".join(profile["warnings"]).lower()
    assert "missing" in joined_warnings
    assert "cross-category" in joined_warnings


def test_render_markdown_contains_only_aggregate_sections(tmp_path):
    matches_path, items_path = _write_fixture(tmp_path)
    profile = build_profile(matches_path, items_path)

    text = render_markdown(profile)

    assert "# E-CUP Matching Human Data Profile" in text
    assert "Category breakdown" in text
    assert "Phone X 128GB" not in text
    assert '{"memory"' not in text
    json.dumps(profile)
