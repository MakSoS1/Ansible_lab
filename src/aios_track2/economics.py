from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, getcontext

import pandas as pd

getcontext().prec = 28


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


def _d(v: object) -> Decimal:
    return Decimal(str(0 if pd.isna(v) else v))


def calculate_npv(monthly: pd.DataFrame, config: EconomicsConfig) -> NpvResult:
    required = {"date", "WOMT_Diff", "WLPT_Diff", "WWIT_Diff", "WLPR"}
    missing = required - set(monthly.columns)
    if missing:
        raise ValueError(f"missing economics columns: {sorted(missing)}")
    frame = monthly.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if (frame["WLPR"].astype(float) > float(config.max_wlpr)).any():
        raise ValueError("WLPR exceeds 500 m3/day contract limit")
    negative = (
        (frame["WLPT_Diff"].astype(float) < 0)
        | (frame["WOMT_Diff"].astype(float) < 0)
        | (frame["WWIT_Diff"].astype(float) < 0)
    )
    excluded = tuple(int(i) for i in frame.index[negative])
    frame = frame.loc[~negative].copy()
    frame = frame[frame["date"] >= pd.Timestamp(config.economic_start)]
    annual_rows: list[dict[str, object]] = []
    total = Decimal("0")
    for year, part in frame.groupby(frame["date"].dt.year, sort=True):
        oil = sum((_d(v) for v in part["WOMT_Diff"]), Decimal("0"))
        liquid = sum((_d(v) for v in part["WLPT_Diff"]), Decimal("0"))
        injection = sum((_d(v) for v in part["WWIT_Diff"]), Decimal("0"))
        gross_margin = oil * (config.oil_price_rub_t - config.oil_deductions_rub_t - config.oil_opex_rub_t)
        variable_opex = liquid * config.liquid_opex_rub_t + injection * config.injection_opex_rub_m3
        days = sum((_d(v) for v in part.get("days", pd.Series([30] * len(part), index=part.index))), Decimal("0"))
        if "well" in part:
            active_well_days = Decimal("0")
            for _, wp in part.groupby("well"):
                active_well_days += sum((_d(v) for v in wp.get("days", pd.Series([30] * len(wp), index=wp.index))), Decimal("0"))
        else:
            active_well_days = days
        fixed_opex = active_well_days / Decimal("365") * config.active_well_rub_year
        event_cost = Decimal("0")
        if "pump_change" in part:
            event_cost += Decimal(int(part["pump_change"].fillna(False).astype(bool).sum())) * config.pump_workover_rub
        if "stop_event" in part:
            event_cost += Decimal(int(part["stop_event"].fillna(False).astype(bool).sum())) * config.stop_or_start_rub
        if "start_event" in part:
            event_cost += Decimal(int(part["start_event"].fillna(False).astype(bool).sum())) * config.stop_or_start_rub
        if "producer_to_injector" in part:
            event_cost += Decimal(int(part["producer_to_injector"].fillna(False).astype(bool).sum())) * config.producer_to_injector_rub
        pre_tax = gross_margin - variable_opex - fixed_opex - event_cost
        tax = max(pre_tax, Decimal("0")) * config.profit_tax
        cashflow = pre_tax - tax
        exponent = Decimal(int(year) - config.economic_start.year)
        discount = (Decimal("1") + config.wacc) ** exponent
        discounted = cashflow / discount
        total += discounted
        annual_rows.append({
            "year": int(year), "oil_t": oil, "liquid_t": liquid, "injection_m3": injection,
            "pre_tax_rub": pre_tax, "tax_rub": tax, "cashflow_rub": cashflow,
            "discount_factor": discount, "discounted_cashflow_rub": discounted,
        })
    annual = pd.DataFrame(annual_rows).set_index("year") if annual_rows else pd.DataFrame()
    return NpvResult(total, excluded, annual)
