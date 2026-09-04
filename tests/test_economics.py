from datetime import date
from decimal import Decimal

import pandas as pd

from aios_track2.economics import EconomicsConfig, calculate_npv


def test_negative_diff_row_is_fully_excluded() -> None:
    monthly = pd.DataFrame(
        [
            {
                "DATA": "2007-01-01",
                "well": "P1",
                "WLPT_Diff": -1.0,
                "WOMT_Diff": 2.0,
                "WWIT_Diff": 0.0,
                "WLPR": 10.0,
                "WWIR": 0.0,
            },
            {
                "DATA": "2007-02-01",
                "well": "P1",
                "WLPT_Diff": 3.0,
                "WOMT_Diff": 2.0,
                "WWIT_Diff": 0.0,
                "WLPR": 10.0,
                "WWIR": 0.0,
            },
        ]
    )
    result = calculate_npv(monthly, EconomicsConfig.default_track2())
    assert len(result.excluded_rows) == 1
    assert result.annual.loc[2007, "oil_t"] == Decimal("2.0")


def test_track2_discount_factor_is_one_in_2007() -> None:
    assert EconomicsConfig.default_track2().discount_factor(date(2007, 12, 31)) == Decimal("1")
