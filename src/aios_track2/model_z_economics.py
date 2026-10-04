from __future__ import annotations

import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .chdd_extract import opm_rows_to_chdd
from .economics_official import REQUIRED_COLUMNS, compute_calculation

TRACK2_ECONOMIC_START = "2007-01-01"
ReportMode = Literal["all", "last_per_month"]


@dataclass(frozen=True, slots=True)
class ModelZDensityMap:
    oil_t_m3: dict[str, float]
    water_t_m3: dict[str, float]
    region_by_well: dict[str, int]
    completion_region_counts: dict[str, dict[int, int]]

    @property
    def mixed_wells(self) -> tuple[str, ...]:
        return tuple(sorted(well for well, counts in self.completion_region_counts.items() if len(counts) > 1))


def _strip_comments(text: str) -> str:
    return re.sub(r"--[^\n]*", "", text)


def _keyword_payload(text: str, keyword: str) -> str:
    clean = _strip_comments(text)
    lines = clean.splitlines()
    start: int | None = None
    for index, raw in enumerate(lines):
        if raw.strip().upper() == keyword.upper():
            start = index + 1
            break
    if start is None:
        raise ValueError(f"keyword {keyword} not found")
    payload: list[str] = []
    keyword_line = re.compile(r"^[A-Z][A-Z0-9_]*$")
    for raw in lines[start:]:
        stripped = raw.strip()
        if payload and keyword_line.fullmatch(stripped):
            break
        payload.append(raw)
    return "\n".join(payload)


def _expand_integer_array(payload: str, expected_size: int) -> np.ndarray:
    values: list[int] = []
    for token in payload.replace("/", " ").split():
        if "*" in token:
            count_text, value_text = token.split("*", 1)
            if not count_text or not value_text:
                raise ValueError(f"unsupported defaulted array token: {token}")
            values.extend([int(value_text)] * int(count_text))
        else:
            values.append(int(token))
    if len(values) != expected_size:
        raise ValueError(f"expected {expected_size} PVTNUM cells, got {len(values)}")
    return np.asarray(values, dtype=np.int16)


def _density_table(payload: str) -> dict[int, tuple[float, float]]:
    table: dict[int, tuple[float, float]] = {}
    for region, record in enumerate(payload.split("/"), start=1):
        tokens = record.split()
        if not tokens:
            continue
        if len(tokens) < 2:
            raise ValueError("DENSITY record must contain oil and water density")
        table[region] = (float(tokens[0]) / 1000.0, float(tokens[1]) / 1000.0)
    if not table:
        raise ValueError("no DENSITY records found")
    return table


def _compdat_cells(text: str) -> dict[str, set[tuple[int, int, int]]]:
    clean = _strip_comments(text)
    cells: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
    in_block = False
    for raw in clean.splitlines():
        stripped = raw.strip()
        upper = stripped.upper()
        if not in_block:
            if upper == "COMPDAT":
                in_block = True
            continue
        if stripped == "/":
            in_block = False
            continue
        if not stripped or not stripped.endswith("/"):
            continue
        tokens = shlex.split(stripped[:-1].strip(), posix=True)
        if len(tokens) < 5:
            continue
        try:
            well = str(tokens[0])
            i, j, k1, k2 = (int(tokens[index]) for index in range(1, 5))
        except ValueError as exc:
            raise ValueError(f"unsupported COMPDAT coordinate record: {stripped}") from exc
        for k in range(k1, k2 + 1):
            cells[well].add((i, j, k))
    if not cells:
        raise ValueError("no COMPDAT completion cells found")
    return dict(cells)


def _single_file(root: Path, suffix: str) -> Path:
    matches = sorted(root.rglob(f"*{suffix}"))
    if len(matches) != 1:
        raise ValueError(f"expected one *{suffix} file, found {len(matches)}")
    return matches[0]


def load_model_z_density_map(root: Path, *, dimensions: tuple[int, int, int] = (91, 102, 59)) -> ModelZDensityMap:
    nx, ny, nz = dimensions
    regs = _single_file(root, "_regs.inc").read_text(encoding="utf-8", errors="strict")
    props = _single_file(root, "_props.inc").read_text(encoding="utf-8", errors="strict")
    schedule = _single_file(root, "_sch.inc").read_text(encoding="utf-8", errors="strict")
    pvtnum = _expand_integer_array(_keyword_payload(regs, "PVTNUM"), nx * ny * nz)
    density_by_region = _density_table(_keyword_payload(props, "DENSITY"))
    completion_cells = _compdat_cells(schedule)

    region_by_well: dict[str, int] = {}
    region_counts_by_well: dict[str, dict[int, int]] = {}
    oil: dict[str, float] = {}
    water: dict[str, float] = {}
    for well, cells in completion_cells.items():
        counts: Counter[int] = Counter()
        for i, j, k in cells:
            if not (1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz):
                raise ValueError(f"completion outside grid for well {well}: {(i, j, k)}")
            flat = (k - 1) * nx * ny + (j - 1) * nx + (i - 1)
            region = int(pvtnum[flat])
            if region <= 0:
                raise ValueError(f"completion maps to inactive/undefined PVTNUM for well {well}: {(i, j, k)}")
            if region not in density_by_region:
                raise ValueError(f"no DENSITY record for PVT region {region}")
            counts[region] += 1
        if not counts:
            raise ValueError(f"no valid completion region for well {well}")
        ordered = counts.most_common()
        if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
            raise ValueError(f"ambiguous dominant PVT region for well {well}: {dict(counts)}")
        region = ordered[0][0]
        region_by_well[well] = region
        region_counts_by_well[well] = dict(sorted(counts.items()))
        oil[well], water[well] = density_by_region[region]
    return ModelZDensityMap(oil, water, region_by_well, region_counts_by_well)


def _report_indices(dates: np.ndarray, report_mode: ReportMode) -> list[int]:
    values = np.asarray(dates).astype(str)
    if report_mode == "all":
        return list(range(len(values)))
    if report_mode == "last_per_month":
        last: dict[str, int] = {}
        for index, value in enumerate(values):
            last[value[:7]] = index
        return [last[month] for month in sorted(last)]
    raise ValueError(f"unsupported report_mode: {report_mode}")


def summary_npz_to_chdd_rows(
    summary_path: Path,
    *,
    oil_density_t_m3: dict[str, float],
    water_density_t_m3: dict[str, float],
    report_mode: ReportMode = "all",
) -> pd.DataFrame:
    with np.load(summary_path) as summary:
        dates = summary["dates"].astype(str)
        wells = summary["wells"].astype(str)
        indices = _report_indices(dates, report_mode)
        columns = {
            "WOPT": summary["well_WOPT"],
            "WWPT": summary["well_WWPT"],
            "WOPR": summary["well_WOPR"],
            "WLPR": summary["well_WLPR"],
            "WWIR": summary["well_WWIR"],
            "WWIT": summary["well_WWIT"],
            "WBHP": summary["well_WBHP"],
            "WTHP": summary["well_WTHP"],
        }
        records: list[dict[str, Any]] = []
        for index in indices:
            for well_index, well in enumerate(wells):
                row: dict[str, Any] = {"DATA": str(dates[index]), "well": str(well)}
                for name, values in columns.items():
                    row[name] = float(values[index, well_index])
                records.append(row)
    opm = pd.DataFrame.from_records(records)
    return opm_rows_to_chdd(opm, oil_density_t_m3=oil_density_t_m3, water_density_t_m3=water_density_t_m3)


def scenario_chdd(
    summary_path: Path,
    *,
    oil_density_t_m3: dict[str, float],
    water_density_t_m3: dict[str, float],
    start_date: str = TRACK2_ECONOMIC_START,
    report_mode: ReportMode = "all",
) -> dict[str, Any]:
    rows = summary_npz_to_chdd_rows(
        summary_path,
        oil_density_t_m3=oil_density_t_m3,
        water_density_t_m3=water_density_t_m3,
        report_mode=report_mode,
    )
    records = rows.loc[:, REQUIRED_COLUMNS].copy()
    records["DATA"] = records["DATA"].dt.strftime("%Y-%m-%d")
    result = compute_calculation(
        records.to_dict(orient="records"),
        headers=REQUIRED_COLUMNS,
        start_date=start_date,
        source_file=str(summary_path),
    )
    result.setdefault("diagnostics", {})["opmReportMode"] = report_mode
    result["diagnostics"]["opmReportRows"] = int(len(rows))
    return result
