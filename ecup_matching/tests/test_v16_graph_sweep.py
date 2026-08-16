"""The sweep must search what v8 never searched, and reproduce its baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecup_matching.ml import run_v16_graph_sweep as sweep


def test_reciprocal_and_transitivity_are_actually_in_the_grid():
    """v8 pinned both reciprocal terms at zero, so they were never evaluated."""
    assert max(sweep.RECIPROCAL_BEST_GRID) > 0.0
    assert max(sweep.RECIPROCAL_TOP3_GRID) > 0.0
    assert max(sweep.SUPPORT_GRID) > 0.0
    assert max(sweep.ORPHAN_GRID) > 0.0
    # The v9 frozen point must remain reachable so the old result is comparable.
    assert 0.02 in sweep.ENDPOINT_RANK_GRID
    assert 0.01 in sweep.AMBIGUITY_GRID
    assert 0.0 in sweep.RECIPROCAL_BEST_GRID


def _fixture(tmp_path: Path, rows: int = 240):
    rng = np.random.default_rng(5)
    cats = ["Электроника", "Обувь", "Одежда"]
    # A graph with real degree: items repeat across pairs.
    left = rng.integers(0, rows // 4, rows)
    right = rng.integers(0, rows // 4, rows) + rows // 4
    category = [cats[i % 3] for i in range(rows // 2)]
    item_ids = list(range(rows // 2))
    items = pd.DataFrame({"id": item_ids, "category": category})
    matches = pd.DataFrame(
        {
            "id1": left,
            "id2": right,
            "target": rng.integers(0, 2, rows),
        }
    )
    # Keep both endpoints inside one category.
    cat_of = dict(zip(items["id"], items["category"]))
    keep = [i for i in range(rows) if cat_of[left[i]] == cat_of[right[i]]]
    matches = matches.iloc[keep].reset_index(drop=True)
    matches.loc[matches.index[0], "target"] = 1
    matches.loc[matches.index[1], "target"] = 0
    items.to_parquet(tmp_path / "items_human.parquet", index=False)
    matches.to_parquet(tmp_path / "matches.parquet", index=False)
    pd.DataFrame(
        {
            "row_index": np.arange(len(matches), dtype=np.int64),
            "fold": np.zeros(len(matches), dtype=np.int16),
            "oof_score": rng.random(len(matches)),
        }
    ).to_parquet(tmp_path / "scores.parquet", index=False)
    return tmp_path


def test_sweep_reports_degree_and_never_opens_gold(tmp_path):
    root = _fixture(tmp_path)
    # Trim the grid so the fixture runs fast; the real grid is asserted above.
    for name, value in (
        ("RECIPROCAL_BEST_GRID", (0.0, 0.02)),
        ("RECIPROCAL_TOP3_GRID", (0.0,)),
        ("ENDPOINT_RANK_GRID", (0.02,)),
        ("AMBIGUITY_GRID", (0.01,)),
        ("SUPPORT_GRID", (0.0, 0.02)),
        ("ORPHAN_GRID", (0.0,)),
    ):
        setattr(sweep, name, value)

    payload = sweep.run_graph_sweep(
        scores_path=root / "scores.parquet",
        human_items_path=root / "items_human.parquet",
        human_matches_path=root / "matches.parquet",
        output_path=root / "out.json",
    )
    assert payload["gold_metric_opened"] is False
    assert payload["gold_rows_scored"] == 0
    assert payload["diagnostic_only"] is True
    assert "degree_report" in payload
    assert payload["degree_report"]["edges"] == payload["rows"]
    assert "fraction_degree_1" in payload["degree_report"]
    saved = json.loads((root / "out.json").read_text(encoding="utf-8"))
    assert saved["best"]["macro_ap"] >= saved["v9_frozen_config_for_reference"]["macro_ap"]


def test_sweep_refuses_a_baseline_it_cannot_reproduce(tmp_path):
    import pytest

    root = _fixture(tmp_path)
    for name, value in (
        ("RECIPROCAL_BEST_GRID", (0.0,)),
        ("RECIPROCAL_TOP3_GRID", (0.0,)),
        ("ENDPOINT_RANK_GRID", (0.02,)),
        ("AMBIGUITY_GRID", (0.01,)),
        ("SUPPORT_GRID", (0.0,)),
        ("ORPHAN_GRID", (0.0,)),
    ):
        setattr(sweep, name, value)
    with pytest.raises(RuntimeError, match="baseline reproduction failed"):
        sweep.run_graph_sweep(
            scores_path=root / "scores.parquet",
            human_items_path=root / "items_human.parquet",
            human_matches_path=root / "matches.parquet",
            output_path=root / "out.json",
            baseline_macro_ap=0.123456,
        )
