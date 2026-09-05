from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


_REQUIRED_COLUMNS = (
    "DATA",
    "well",
    "WOPT",
    "WWPT",
    "WOPR",
    "WLPR",
    "WWIR",
    "WWIT",
    "WBHP",
    "WTHP",
)


def _require_columns(rows: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"missing required OPM columns: {', '.join(missing)}")


def _density_series(
    wells: pd.Series,
    density_by_well: Mapping[str, float],
    *,
    phase: str,
) -> pd.Series:
    missing = sorted({str(well) for well in wells.astype(str).unique()} - set(density_by_well))
    if missing:
        raise ValueError(f"missing {phase} density for wells: {', '.join(missing)}")
    density = wells.astype(str).map(density_by_well).astype(float)
    if density.isna().any() or (density <= 0.0).any():
        bad = sorted(wells[density.isna() | (density <= 0.0)].astype(str).unique())
        raise ValueError(f"invalid {phase} density for wells: {', '.join(bad)}")
    return density


def _cumulative_delta(frame: pd.DataFrame, column: str) -> pd.Series:
    delta = frame.groupby("well", sort=False)[column].diff()
    return delta.where(delta.notna(), frame[column])


def opm_rows_to_chdd(
    rows: pd.DataFrame,
    *,
    oil_density_t_m3: Mapping[str, float],
    water_density_t_m3: Mapping[str, float],
) -> pd.DataFrame:
    """Convert well-level OPM summary rows to the organizer CHDD input contract.

    OPM oil/water cumulative vectors are surface volumes in m3.  The supplied
    CHDD implementation expects cumulative oil and liquid in tonnes, while
    liquid/injection rates and injected cumulative water remain volumetric.
    Densities are therefore explicit per-well inputs and never fall back to a
    global average.
    """
    _require_columns(rows)
    result = rows.copy()
    result["DATA"] = pd.to_datetime(result["DATA"], errors="raise")
    result["well"] = result["well"].astype(str)
    result = result.sort_values(["well", "DATA"], kind="stable").reset_index(drop=True)

    oil_density = _density_series(result["well"], oil_density_t_m3, phase="oil")
    water_density = _density_series(result["well"], water_density_t_m3, phase="water")

    result["WOMT"] = result["WOPT"].astype(float) * oil_density
    result["WOMR"] = result["WOPR"].astype(float) * oil_density
    result["WLPT"] = (
        result["WOPT"].astype(float) * oil_density
        + result["WWPT"].astype(float) * water_density
    )

    result["WLPR"] = result["WLPR"].astype(float)
    result["WWIR"] = result["WWIR"].astype(float)
    result["WWIT"] = result["WWIT"].astype(float)
    result["BHP"] = result["WBHP"].astype(float)
    result["THP"] = result["WTHP"].astype(float)
    result["WEFF"] = 1.0

    result["WOMT_Diff"] = _cumulative_delta(result, "WOMT")
    result["WLPT_Diff"] = _cumulative_delta(result, "WLPT")
    result["WWIT_Diff"] = _cumulative_delta(result, "WWIT")
    return result
