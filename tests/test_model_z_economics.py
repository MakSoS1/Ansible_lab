from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aios_track2.model_z_economics import (
    load_model_z_density_map,
    scenario_chdd,
    summary_npz_to_chdd_rows,
)


def _write_tiny_model(root: Path) -> None:
    (root / "Tiny_regs.inc").write_text("PVTNUM\n 1 2 1 2 /\n", encoding="utf-8")
    (root / "Tiny_props.inc").write_text(
        "DENSITY\n 900 1100 1 /\n 950 1200 1 /\n",
        encoding="utf-8",
    )
    (root / "Tiny_sch.inc").write_text(
        "COMPDAT\n"
        " 'A' 1 1 1 1 OPEN 1* 1* 0.1 1* 0 1* Z /\n"
        " 'B' 2 1 1 2 OPEN 1* 1* 0.1 1* 0 1* Z /\n"
        "/\n",
        encoding="utf-8",
    )


def test_density_map_uses_completion_pvt_region_not_global_average(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path)
    density = load_model_z_density_map(tmp_path, dimensions=(2, 1, 2))
    assert density.region_by_well == {"A": 1, "B": 2}
    assert density.oil_t_m3 == {"A": pytest.approx(0.9), "B": pytest.approx(0.95)}
    assert density.water_t_m3 == {"A": pytest.approx(1.1), "B": pytest.approx(1.2)}


def _write_summary(path: Path) -> None:
    dates = np.asarray(["2006-12-01", "2007-01-01", "2007-01-15", "2007-02-01"])
    wells = np.asarray(["A"])
    shape = (len(dates), 1)
    np.savez_compressed(
        path,
        dates=dates,
        wells=wells,
        well_WOPT=np.asarray([[10.0], [12.0], [13.0], [15.0]]),
        well_WWPT=np.asarray([[20.0], [21.0], [22.0], [23.0]]),
        well_WWIT=np.asarray([[30.0], [31.0], [32.0], [34.0]]),
        well_WOPR=np.full(shape, 2.0),
        well_WWPR=np.full(shape, 3.0),
        well_WWIR=np.full(shape, 4.0),
        well_WLPR=np.full(shape, 5.0),
        well_WBHP=np.full(shape, 100.0),
        well_WTHP=np.full(shape, 50.0),
        well_WWCT=np.full(shape, 0.6),
    )


def test_summary_conversion_keeps_last_report_per_calendar_month(tmp_path: Path) -> None:
    summary = tmp_path / "summary.npz"
    _write_summary(summary)
    rows = summary_npz_to_chdd_rows(summary, oil_density_t_m3={"A": 0.9}, water_density_t_m3={"A": 1.1})
    assert rows["DATA"].dt.strftime("%Y-%m-%d").tolist() == ["2006-12-01", "2007-01-15", "2007-02-01"]
    january = rows.iloc[1]
    assert january["WOMT"] == pytest.approx(13.0 * 0.9)
    assert january["WOMT_Diff"] == pytest.approx((13.0 - 10.0) * 0.9)
    assert january["WWIT_Diff"] == pytest.approx(2.0)


def test_scenario_chdd_uses_track2_economic_start(tmp_path: Path) -> None:
    summary = tmp_path / "summary.npz"
    _write_summary(summary)
    result = scenario_chdd(summary, oil_density_t_m3={"A": 0.9}, water_density_t_m3={"A": 1.1})
    assert result["version"] == "7.0.2-negative-row-filter"
    assert result["startDate"] == "2007-01-01"
