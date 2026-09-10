from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from aios_track2.config import load_config
from aios_track2.dataset import split_scenarios
from aios_track2.economics import EconomicsConfig, calculate_npv
from aios_track2.schedule import Control, Schedule, project_schedule, write_schedule_text


def test_base_config_has_track2_contract() -> None:
    cfg = load_config(Path("configs/base.yaml"))
    assert cfg.seed == 42
    assert cfg.economic_start == date(2007, 1, 1)
    assert cfg.max_wlpr_m3_day == 500.0
    assert cfg.hf_dataset_id == "MakSoS1/aios-track2-runs"


def test_wlpr_above_limit_is_rejected() -> None:
    schedule = Schedule(
        [Control(date=date(2007, 1, 1), well="P1", role="PRODUCER", liquid_rate=501.0)]
    )
    result = project_schedule(schedule)
    assert result.accepted is False
    assert any(v.code == "WLPR_LIMIT" for v in result.violations)


def test_reverse_conversion_is_rejected() -> None:
    schedule = Schedule(
        [
            Control(date=date(2007, 1, 1), well="P1", role="PRODUCER", liquid_rate=100),
            Control(
                date=date(2007, 4, 1),
                well="P1",
                role="INJECTOR",
                injection_rate=100,
                converted_to_injector=True,
            ),
            Control(date=date(2007, 7, 1), well="P1", role="PRODUCER", liquid_rate=100),
        ]
    )
    result = project_schedule(schedule)
    assert any(v.code == "REVERSE_CONVERSION" for v in result.violations)


def test_wconprod_lrat_is_rendered_in_item_7() -> None:
    schedule = Schedule(
        [Control(date=date(2007, 1, 1), well="P1", role="PRODUCER", liquid_rate=250, bhp=120)]
    )
    text = write_schedule_text(schedule)
    record = next(line.strip() for line in text.splitlines() if line.strip().startswith("'P1'"))
    tokens = record.replace("/", "").split()
    assert tokens[:3] == ["'P1'", "'OPEN'", "'LRAT'"]
    assert tokens[3] == "3*"
    assert tokens[4] == "250.000"
    assert tokens[5] == "1*"
    assert tokens[6] == "120.000"


def test_scenario_split_never_leaks_rows_between_partitions() -> None:
    frame = pd.DataFrame(
        {
            "scenario_id": [f"s{i}" for i in range(10) for _ in range(3)],
            "value": range(30),
        }
    )
    split = split_scenarios(frame, seed=7)
    ids = {name: set(part["scenario_id"]) for name, part in split.items()}
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])
    assert set.union(*ids.values()) == {f"s{i}" for i in range(10)}


def test_negative_diff_row_is_fully_excluded() -> None:
    monthly = pd.DataFrame(
        [
            {
                "date": "2007-01-01",
                "well": "P1",
                "WLPT_Diff": -1.0,
                "WOMT_Diff": 2.0,
                "WWIT_Diff": 0.0,
                "WLPR": 10.0,
                "days": 31,
            },
            {
                "date": "2007-02-01",
                "well": "P1",
                "WLPT_Diff": 3.0,
                "WOMT_Diff": 2.0,
                "WWIT_Diff": 0.0,
                "WLPR": 10.0,
                "days": 28,
            },
        ]
    )
    result = calculate_npv(monthly, EconomicsConfig())
    assert result.excluded_rows == (0,)
    assert result.annual.loc[2007, "oil_t"] == Decimal("2.0")


def test_economics_rejects_wlpr_contract_violation() -> None:
    monthly = pd.DataFrame(
        [
            {
                "date": "2007-01-01",
                "well": "P1",
                "WLPT_Diff": 10.0,
                "WOMT_Diff": 2.0,
                "WWIT_Diff": 0.0,
                "WLPR": 500.1,
            }
        ]
    )
    with pytest.raises(ValueError, match="WLPR"):
        calculate_npv(monthly, EconomicsConfig())
