from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from aios_track2.vendor import chdd_model


@dataclass(frozen=True)
class EconomicsConfig:
    start: date
    oil_price: Decimal = Decimal("28000")
    wacc: Decimal = Decimal("0.10")

    @classmethod
    def default_track2(cls) -> EconomicsConfig:
        return cls(start=date(2007, 1, 1))

    @classmethod
    def historic_1991(cls) -> EconomicsConfig:
        return cls(start=date(1991, 1, 1))

    def discount_factor(self, when: date) -> Decimal:
        years = max(0, when.year - self.start.year)
        return Decimal("1") / ((Decimal("1") + self.wacc) ** years)


@dataclass(frozen=True)
class RowReference:
    well: str
    data: str


@dataclass(frozen=True)
class NpvResult:
    npv_mrub: Decimal
    annual: pd.DataFrame
    excluded_rows: tuple[RowReference, ...]
    monthly: pd.DataFrame
    method: str
    backend: str = "chdd_model"
    raw: dict | None = None


def _records_from_frame(monthly: pd.DataFrame) -> list[dict]:
    frame = monthly.copy()
    if "DATA" not in frame.columns and "date" in frame.columns:
        frame["DATA"] = frame["date"]
    records = frame.to_dict(orient="records")
    if records:
        required = set(chdd_model.REQUIRED_COLUMNS)
        for record in records:
            for column in required:
                record.setdefault(column, 0.0)
    return records


def calculate_npv(monthly: pd.DataFrame, config: EconomicsConfig | None = None) -> NpvResult:
    config = config or EconomicsConfig.default_track2()
    records = _records_from_frame(monthly)
    excluded = tuple(
        RowReference(well=str(row.get("well")), data=str(chdd_model.normalize_date(row.get("DATA"))))
        for row in records
        if chdd_model.to_number(row.get("WLPT_Diff")) < 0
        or chdd_model.to_number(row.get("WOMT_Diff")) < 0
        or chdd_model.to_number(row.get("WWIT_Diff")) < 0
    )
    payload = chdd_model.compute_calculation(
        records,
        headers=chdd_model.REQUIRED_COLUMNS,
        start_date=config.start.isoformat(),
        name="AIOS Track 2",
    )
    annual_rows = []
    for item in payload.get("annual", []):
        annual_rows.append(
            {
                "year": int(item["year"]),
                "oil_t": Decimal(str(item.get("oilT", item.get("oilKt", 0.0) * 1000))),
                "npv_mrub": Decimal(str(item.get("chddM", 0.0))),
                "fcf_mrub": Decimal(str(item.get("fcfM", 0.0))),
            }
        )
    annual = pd.DataFrame(annual_rows).set_index("year") if annual_rows else pd.DataFrame(columns=["oil_t", "npv_mrub"])
    summary = payload.get("summary", {})
    npv = Decimal(str(summary.get("totalChddM", 0.0)))
    return NpvResult(
        npv_mrub=npv,
        annual=annual,
        excluded_rows=excluded,
        monthly=pd.DataFrame(payload.get("fieldMonthly", [])),
        method=f"start={config.start.isoformat()}",
        raw=payload,
    )


def calculate_npv_from_path(path: Path, economic_start: date) -> NpvResult:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)
    return calculate_npv(frame, EconomicsConfig(start=economic_start))
