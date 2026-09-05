from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True, slots=True)
class ScheduleStructure:
    explicit_dates: tuple[date, ...]
    tstep_days: tuple[float, ...]
    producer_blocks: int
    injector_blocks: int
    producer_records: int
    injector_records: int
    producer_modes: tuple[str, ...]
    injector_modes: tuple[str, ...]


def _strip_comments(text: str) -> str:
    return re.sub(r"--[^\n]*", "", text)


def _blocks(clean: str, keyword: str) -> list[str]:
    lines = clean.splitlines()
    out: list[str] = []
    current: list[str] | None = None
    for raw in lines:
        line = raw.strip()
        upper = line.upper()
        if current is None:
            if upper == keyword:
                current = []
            continue
        if line == "/":
            out.append("\n".join(current))
            current = None
            continue
        current.append(line)
    return out


def _records(block: str) -> list[str]:
    joined = " ".join(line for line in block.splitlines() if line.strip())
    return [piece.strip() for piece in joined.split("/") if piece.strip()]


def _parse_dates(clean: str) -> tuple[date, ...]:
    parsed: list[date] = []
    for block in _blocks(clean, "DATES"):
        for record in _records(block):
            token = record.replace("'", "").replace('"', "").strip()
            for fmt in ("%d %b %Y", "%d %B %Y"):
                try:
                    from datetime import datetime

                    parsed.append(datetime.strptime(token, fmt).date())
                    break
                except ValueError:
                    continue
    return tuple(parsed)


def _parse_tsteps(clean: str) -> tuple[float, ...]:
    values: list[float] = []
    for block in _blocks(clean, "TSTEP"):
        for record in _records(block):
            for token in record.split():
                m = re.fullmatch(r"(?:(\d+)\*)?([-+]?\d+(?:\.\d+)?)", token)
                if not m:
                    continue
                repeat = int(m.group(1) or 1)
                values.extend([float(m.group(2))] * repeat)
    return tuple(values)


def _well_mode_stats(clean: str, keyword: str, mode_index: int) -> tuple[int, int, tuple[str, ...]]:
    blocks = _blocks(clean, keyword)
    records = [record for block in blocks for record in _records(block)]
    modes: set[str] = set()
    for record in records:
        tokens = re.findall(r"'[^']*'|\S+", record)
        if len(tokens) > mode_index:
            modes.add(tokens[mode_index].strip("'\"").upper())
    return len(blocks), len(records), tuple(sorted(modes))


def inspect_schedule_text(text: str) -> ScheduleStructure:
    clean = _strip_comments(text)
    prod_blocks, prod_records, prod_modes = _well_mode_stats(clean, "WCONPROD", 2)
    inj_blocks, inj_records, inj_modes = _well_mode_stats(clean, "WCONINJE", 3)
    return ScheduleStructure(
        explicit_dates=_parse_dates(clean),
        tstep_days=_parse_tsteps(clean),
        producer_blocks=prod_blocks,
        injector_blocks=inj_blocks,
        producer_records=prod_records,
        injector_records=inj_records,
        producer_modes=prod_modes,
        injector_modes=inj_modes,
    )


def advance_by_tsteps(start: date, tstep_days: tuple[float, ...]) -> tuple[date, ...]:
    current = start
    out: list[date] = []
    for days in tstep_days:
        current = current + timedelta(days=days)
        out.append(current)
    return tuple(out)
