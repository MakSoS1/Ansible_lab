from __future__ import annotations

import pandas as pd

from aios_track2.chdd_extract import opm_rows_to_chdd


def test_opm_rows_to_chdd_converts_surface_volumes_to_mass_by_well_density() -> None:
    rows = pd.DataFrame(
        [
            {
                "DATA": "2007-01-01",
                "well": "P1",
                "WOPT": 100.0,
                "WWPT": 50.0,
                "WOPR": 10.0,
                "WLPR": 15.0,
                "WWIR": 0.0,
                "WWIT": 0.0,
                "WBHP": 120.0,
                "WTHP": 90.0,
            },
            {
                "DATA": "2007-02-01",
                "well": "P1",
                "WOPT": 140.0,
                "WWPT": 70.0,
                "WOPR": 8.0,
                "WLPR": 13.0,
                "WWIR": 0.0,
                "WWIT": 0.0,
                "WBHP": 118.0,
                "WTHP": 89.0,
            },
            {
                "DATA": "2007-01-01",
                "well": "P2",
                "WOPT": 50.0,
                "WWPT": 25.0,
                "WOPR": 5.0,
                "WLPR": 8.0,
                "WWIR": 0.0,
                "WWIT": 0.0,
                "WBHP": 125.0,
                "WTHP": 91.0,
            },
        ]
    )
    result = opm_rows_to_chdd(
        rows,
        oil_density_t_m3={"P1": 0.9, "P2": 0.92},
        water_density_t_m3={"P1": 1.1, "P2": 1.12},
    )

    p1_jan = result[(result["well"] == "P1") & (result["DATA"] == pd.Timestamp("2007-01-01"))].iloc[0]
    p1_feb = result[(result["well"] == "P1") & (result["DATA"] == pd.Timestamp("2007-02-01"))].iloc[0]
    p2_jan = result[(result["well"] == "P2") & (result["DATA"] == pd.Timestamp("2007-01-01"))].iloc[0]

    assert p1_jan["WOMT"] == 90.0
    assert p1_jan["WOMR"] == 9.0
    assert p1_jan["WLPT"] == 145.0
    assert p1_jan["WLPR"] == 15.0
    assert p1_jan["WOMT_Diff"] == 90.0
    assert p1_jan["WLPT_Diff"] == 145.0
    assert p1_feb["WOMT"] == 126.0
    assert p1_feb["WLPT"] == 203.0
    assert p1_feb["WOMT_Diff"] == 36.0
    assert p1_feb["WLPT_Diff"] == 58.0
    assert p2_jan["WOMT"] == 46.0
    assert p2_jan["WLPT"] == 74.0
    assert p1_feb["BHP"] == 118.0
    assert p1_feb["THP"] == 89.0
    assert p1_feb["WEFF"] == 1.0


def test_opm_rows_to_chdd_preserves_injection_volume_and_computes_delta() -> None:
    rows = pd.DataFrame(
        [
            {
                "DATA": "2007-01-01",
                "well": "I1",
                "WOPT": 0.0,
                "WWPT": 0.0,
                "WOPR": 0.0,
                "WLPR": 0.0,
                "WWIR": 100.0,
                "WWIT": 1000.0,
                "WBHP": 180.0,
                "WTHP": 130.0,
            },
            {
                "DATA": "2007-02-01",
                "well": "I1",
                "WOPT": 0.0,
                "WWPT": 0.0,
                "WOPR": 0.0,
                "WLPR": 0.0,
                "WWIR": 110.0,
                "WWIT": 4100.0,
                "WBHP": 181.0,
                "WTHP": 131.0,
            },
        ]
    )
    result = opm_rows_to_chdd(rows, oil_density_t_m3={"I1": 0.9}, water_density_t_m3={"I1": 1.1})
    jan, feb = result.sort_values("DATA").to_dict("records")
    assert jan["WWIR"] == 100.0
    assert jan["WWIT"] == 1000.0
    assert jan["WWIT_Diff"] == 1000.0
    assert feb["WWIR"] == 110.0
    assert feb["WWIT"] == 4100.0
    assert feb["WWIT_Diff"] == 3100.0


def test_opm_rows_to_chdd_rejects_missing_density_instead_of_silent_global_fallback() -> None:
    rows = pd.DataFrame(
        [
            {
                "DATA": "2007-01-01",
                "well": "P1",
                "WOPT": 1.0,
                "WWPT": 0.0,
                "WOPR": 1.0,
                "WLPR": 1.0,
                "WWIR": 0.0,
                "WWIT": 0.0,
                "WBHP": 100.0,
                "WTHP": 80.0,
            }
        ]
    )
    try:
        opm_rows_to_chdd(rows, oil_density_t_m3={}, water_density_t_m3={"P1": 1.0})
    except ValueError as exc:
        assert "density" in str(exc).lower()
        assert "P1" in str(exc)
    else:
        raise AssertionError("missing per-well density must fail closed")
