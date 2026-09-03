from __future__ import annotations

from calendar import monthrange
from datetime import date

import numpy as np
import pandas as pd

from aios_track2.schedule import Control, Schedule, WellRole

REQUIRED_COLUMNS = [
    "DATA",
    "well",
    "WLPT",
    "WLPR",
    "WOMT",
    "WOMR",
    "WWIR",
    "WWIT",
    "THP",
    "BHP",
    "WEFF",
    "WLPT_Diff",
    "WOMT_Diff",
    "WWIT_Diff",
]


def _days_in_month(value: date) -> int:
    return monthrange(value.year, value.month)[1]


class ProxyFlow:
    """Reduced-order waterflood proxy used when OPM Flow is unavailable.

    The proxy is intentionally simple: liquid offtake, water-cut growth from
    nearby injection, and a voidage-replacement pressure term. It is not a
    substitute for OPM on the official leaderboard, but it makes surrogate
    training, CEM/CMA-ES and MAPPO comparable on Apple Silicon runners.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run(self, schedule: Schedule) -> pd.DataFrame:
        return proxy_monthly(schedule, seed=self.seed)


def proxy_monthly(schedule: Schedule, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wells = sorted({control.well for control in schedule.controls})
    well_index = {name: index for index, name in enumerate(wells)}
    n_wells = max(len(wells), 1)
    affinity = rng.normal(0.0, 0.15, size=(n_wells, n_wells))
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)
    wct = {name: float(0.15 + 0.05 * (index % 5)) for index, name in enumerate(wells)}
    pressure = 180.0
    cumulative = {name: {"oil": 0.0, "liq": 0.0, "inj": 0.0} for name in wells}
    rows: list[dict[str, float | str]] = []
    by_date: dict[date, list[Control]] = {}
    for control in schedule.sorted().controls:
        by_date.setdefault(control.date, []).append(control)

    for step, (day, controls) in enumerate(sorted(by_date.items()), start=1):
        days = _days_in_month(day)
        injection = 0.0
        offtake = 0.0
        inj_by_well = {name: 0.0 for name in wells}
        liq_by_well = {name: 0.0 for name in wells}
        for control in controls:
            if control.status != "OPEN":
                continue
            if control.role == WellRole.INJECTOR or control.wwir > 0:
                inj_by_well[control.well] = min(control.wwir, 500.0)
                injection += inj_by_well[control.well]
            else:
                liq_by_well[control.well] = min(control.wlpr, 500.0)
                offtake += liq_by_well[control.well]
        voidage = injection / max(offtake, 1.0)
        pressure = max(80.0, min(260.0, pressure + 4.0 * (voidage - 1.0) - 0.4))
        for name in wells:
            idx = well_index[name]
            neighbor_inj = float(affinity[idx] @ np.array([inj_by_well[other] for other in wells]))
            wct[name] = min(0.95, max(0.05, 0.92 * wct[name] + 0.0008 * neighbor_inj + 0.004 * step / 12))
            liquid = liq_by_well[name]
            oil_rate = liquid * (1.0 - wct[name]) * (pressure / 180.0)
            inj_rate = inj_by_well[name]
            oil_month = oil_rate * days
            liq_month = liquid * days
            inj_month = inj_rate * days
            cumulative[name]["oil"] += oil_month
            cumulative[name]["liq"] += liq_month
            cumulative[name]["inj"] += inj_month
            rows.append(
                {
                    "DATA": day.isoformat(),
                    "well": name,
                    "WLPT": cumulative[name]["liq"],
                    "WLPR": liquid,
                    "WOMT": cumulative[name]["oil"],
                    "WOMR": oil_rate,
                    "WWIR": inj_rate,
                    "WWIT": cumulative[name]["inj"],
                    "THP": 20.0,
                    "BHP": pressure - 8.0 * wct[name],
                    "WEFF": 1.0 if liquid > 0 or inj_rate > 0 else 0.0,
                    "WLPT_Diff": liq_month,
                    "WOMT_Diff": oil_month,
                    "WWIT_Diff": inj_month,
                    "WCT": wct[name],
                }
            )
    frame = pd.DataFrame(rows)
    for column in REQUIRED_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    return frame[REQUIRED_COLUMNS + [column for column in frame.columns if column not in REQUIRED_COLUMNS]]
