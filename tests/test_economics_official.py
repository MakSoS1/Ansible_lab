from datetime import date
from decimal import Decimal

import pandas as pd

from aios_track2.economics import OFFICIAL_CHDD_VERSION, EconomicsConfig, calculate_npv


def test_organizer_chdd_version_is_locked() -> None:
    assert OFFICIAL_CHDD_VERSION == "7.0.2-negative-row-filter"


def test_official_stop_start_conversion_parity() -> None:
    rows = [
        {"date": "2014-01-01", "well": "A", "WLPR": 100, "WOMR": 10, "WLPT_Diff": 1000, "WOMT_Diff": 400, "WWIT_Diff": 0},
        {"date": "2014-02-01", "well": "A", "WLPR": 0, "WOMR": 0, "WLPT_Diff": 0, "WOMT_Diff": 0, "WWIT_Diff": 0},
        {"date": "2014-03-01", "well": "A", "WLPR": 100, "WOMR": 10, "WLPT_Diff": 1000, "WOMT_Diff": 400, "WWIT_Diff": 0},
        {"date": "2014-01-01", "well": "B", "WLPR": 100, "WOMR": 10, "WLPT_Diff": 1000, "WOMT_Diff": 400, "WWIT_Diff": 0},
        {"date": "2014-02-01", "well": "B", "WLPR": 0, "WOMR": 0, "WWIR": 80, "WLPT_Diff": 0, "WOMT_Diff": 0, "WWIT_Diff": 2000},
    ]
    result = calculate_npv(pd.DataFrame(rows), EconomicsConfig(economic_start=date(2014, 1, 1)))
    annual = result.official["annual"][0]
    assert annual["stopCount"] == 1
    assert annual["startCount"] == 1
    assert annual["startStopCount"] == 2
    assert annual["conversionCount"] == 1
    assert annual["startStopCostM"] == 2.0
    assert annual["conversionOpexM"] == 5.0
    assert abs(result.official["summary"]["totalChddM"] - 1.7539999999999978) < 1e-12
    assert result.npv_rub == Decimal("1753999.9999999978")


def test_official_negative_rows_are_excluded_before_events_and_economics() -> None:
    rows = [
        {"date": "2014-01-01", "well": "A", "WLPR": 100, "WLPT_Diff": 200, "WOMT_Diff": 100, "WWIT_Diff": 0},
        {"date": "2014-02-01", "well": "A", "WLPR": 100, "WLPT_Diff": 200, "WOMT_Diff": -40, "WWIT_Diff": 0},
        {"date": "2014-03-01", "well": "A", "WLPR": 100, "WLPT_Diff": -10, "WOMT_Diff": 20, "WWIT_Diff": 0},
        {"date": "2014-04-01", "well": "A", "WWIR": 50, "WLPR": 0, "WLPT_Diff": 0, "WOMT_Diff": 0, "WWIT_Diff": -10},
    ]
    result = calculate_npv(pd.DataFrame(rows), EconomicsConfig(economic_start=date(2014, 1, 1)))
    assert result.excluded_rows == (1, 2, 3)
    assert result.official["summary"]["totalOilKt"] == 0.1
    assert result.official["diagnostics"]["excludedNegativeRows"] == 3
