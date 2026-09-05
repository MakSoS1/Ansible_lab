from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, getcontext

import pandas as pd

from .economics_official import DEFAULT_ASSUMPTIONS, DEFAULT_PUMPS, REQUIRED_COLUMNS, VERSION, compute_calculation

getcontext().prec = 28
OFFICIAL_CHDD_VERSION = VERSION


@dataclass(frozen=True, slots=True)
class EconomicsConfig:
    economic_start: date = date(2007, 1, 1)
    oil_price_rub_t: Decimal = Decimal("28000")
    oil_deductions_rub_t: Decimal = Decimal("19600")
    oil_opex_rub_t: Decimal = Decimal("40")
    liquid_opex_rub_t: Decimal = Decimal("100")
    injection_opex_rub_m3: Decimal = Decimal("30")
    active_well_rub_year: Decimal = Decimal("1000000")
    pump_workover_rub: Decimal = Decimal("1800000")
    stop_or_start_rub: Decimal = Decimal("1000000")
    producer_to_injector_rub: Decimal = Decimal("5000000")
    profit_tax: Decimal = Decimal("0.25")
    property_tax: Decimal = Decimal("0.022")
    wacc: Decimal = Decimal("0.10")
    max_wlpr: Decimal = Decimal("500")


@dataclass(frozen=True, slots=True)
class NpvResult:
    npv_rub: Decimal
    excluded_rows: tuple[int, ...]
    annual: pd.DataFrame
    official: dict[str, object]


def _float(value: Decimal) -> float:
    return float(value)


def _assumptions(config: EconomicsConfig) -> dict[str, object]:
    values = dict(DEFAULT_ASSUMPTIONS)
    values.update(
        {
            "oilPriceRubT": _float(config.oil_price_rub_t),
            "deductionsRubT": _float(config.oil_deductions_rub_t),
            "oilOpexRubT": _float(config.oil_opex_rub_t),
            "liquidOpexRubT": _float(config.liquid_opex_rub_t),
            "injectionOpexRubM3": _float(config.injection_opex_rub_m3),
            "fundAnnualRubWell": _float(config.active_well_rub_year),
            "pumpOperationCostM": _float(config.pump_workover_rub) / 1_000_000,
            "stopStartCostM": _float(config.stop_or_start_rub) / 1_000_000,
            "conversionBaseCostM": _float(config.producer_to_injector_rub) / 1_000_000,
            "profitTaxRate": _float(config.profit_tax) * 100,
            "propertyTaxRate": _float(config.property_tax) * 100,
            "waccRate": _float(config.wacc) * 100,
        }
    )
    return values


def _records(monthly: pd.DataFrame) -> list[dict[str, object]]:
    frame = monthly.copy()
    if "DATA" not in frame and "date" in frame:
        frame["DATA"] = frame["date"]
    if "well" not in frame:
        frame["well"] = "FIELD"
    for column in REQUIRED_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    frame["DATA"] = pd.to_datetime(frame["DATA"]).dt.strftime("%Y-%m-%d")
    frame["well"] = frame["well"].astype(str)
    return frame[REQUIRED_COLUMNS].where(pd.notna(frame[REQUIRED_COLUMNS]), 0).to_dict(orient="records")


def calculate_npv(monthly: pd.DataFrame, config: EconomicsConfig) -> NpvResult:
    required = {"WOMT_Diff", "WLPT_Diff", "WWIT_Diff", "WLPR"}
    missing = required - set(monthly.columns)
    if missing:
        raise ValueError(f"missing economics columns: {sorted(missing)}")
    if (monthly["WLPR"].astype(float) > float(config.max_wlpr)).any():
        raise ValueError("WLPR exceeds 500 m3/day contract limit")
    negative = (
        (monthly["WLPT_Diff"].astype(float) < 0)
        | (monthly["WOMT_Diff"].astype(float) < 0)
        | (monthly["WWIT_Diff"].astype(float) < 0)
    )
    excluded = tuple(int(i) for i in monthly.index[negative])
    official = compute_calculation(
        _records(monthly),
        headers=REQUIRED_COLUMNS,
        assumptions=_assumptions(config),
        pumps=DEFAULT_PUMPS,
        start_date=config.economic_start.isoformat(),
        name="AIOS Track 2 contract NPV",
    )
    annual = pd.DataFrame(official["annual"])
    if not annual.empty:
        annual["oil_t"] = annual["oilKt"].map(lambda value: Decimal(str(value * 1000)))
        annual["liquid_t"] = annual["liquidKt"].map(lambda value: Decimal(str(value * 1000)))
        annual["injection_m3"] = annual["injectionKm3"].map(lambda value: Decimal(str(value * 1000)))
        annual = annual.set_index("year")
    npv_rub = Decimal(str(official["summary"]["totalChddM"])) * Decimal("1000000")
    return NpvResult(npv_rub=npv_rub, excluded_rows=excluded, annual=annual, official=official)
