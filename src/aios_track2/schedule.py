from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Literal

Role = Literal["PRODUCER", "INJECTOR"]
Status = Literal["OPEN", "SHUT"]


@dataclass(frozen=True, slots=True)
class Control:
    date: date
    well: str
    role: Role
    status: Status = "OPEN"
    liquid_rate: float | None = None
    injection_rate: float | None = None
    bhp: float | None = None
    thp: float | None = None
    converted_to_injector: bool = False


@dataclass(frozen=True, slots=True)
class Schedule:
    controls: tuple[Control, ...]

    def __init__(self, controls: Iterable[Control]):
        object.__setattr__(self, "controls", tuple(sorted(controls, key=lambda c: (c.date, c.well))))


@dataclass(frozen=True, slots=True)
class Violation:
    code: str
    well: str
    at: date
    message: str


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    schedule: Schedule
    violations: tuple[Violation, ...]

    @property
    def accepted(self) -> bool:
        return not self.violations


def project_schedule(
    schedule: Schedule,
    *,
    known_wells: set[str] | None = None,
    max_wlpr: float = 500.0,
    max_injection_bhp: float | None = None,
) -> ProjectionResult:
    violations: list[Violation] = []
    seen: set[tuple[date, str]] = set()
    roles: dict[str, Role] = {}
    for c in schedule.controls:
        key = (c.date, c.well)
        if key in seen:
            violations.append(Violation("DUPLICATE_CONTROL", c.well, c.date, "duplicate date/well control"))
        seen.add(key)
        if known_wells is not None and c.well not in known_wells:
            violations.append(Violation("UNKNOWN_WELL", c.well, c.date, "well is not present in Model Z"))
        for value, code in ((c.liquid_rate, "NEGATIVE_LIQUID"), (c.injection_rate, "NEGATIVE_INJECTION")):
            if value is not None and value < 0:
                violations.append(Violation(code, c.well, c.date, "rate cannot be negative"))
        if c.liquid_rate is not None and c.liquid_rate > max_wlpr:
            violations.append(Violation("WLPR_LIMIT", c.well, c.date, f"WLPR {c.liquid_rate} > {max_wlpr}"))
        if c.role == "INJECTOR" and max_injection_bhp is not None and c.bhp is not None and c.bhp > max_injection_bhp:
            violations.append(Violation("INJECTION_BHP_LIMIT", c.well, c.date, "injection BHP exceeds configured limit"))
        previous = roles.get(c.well)
        if previous == "INJECTOR" and c.role == "PRODUCER":
            violations.append(Violation("REVERSE_CONVERSION", c.well, c.date, "injector cannot be converted back to producer"))
        if previous == "PRODUCER" and c.role == "INJECTOR" and not c.converted_to_injector:
            violations.append(Violation("UNDECLARED_CONVERSION", c.well, c.date, "producer-to-injector transition must be explicit"))
        roles[c.well] = c.role
    return ProjectionResult(schedule, tuple(sorted(violations, key=lambda v: (v.at, v.well, v.code))))


def _date_token(d: date) -> str:
    return f"{d.day} {d.strftime('%b').upper()} {d.year}"


def write_schedule_text(schedule: Schedule) -> str:
    chunks: list[str] = []
    by_date: dict[date, list[Control]] = {}
    for c in schedule.controls:
        by_date.setdefault(c.date, []).append(c)
    for d in sorted(by_date):
        chunks.extend(["DATES", f"  {_date_token(d)} /", "/"])
        producers = [c for c in by_date[d] if c.role == "PRODUCER" and c.status == "OPEN"]
        injectors = [c for c in by_date[d] if c.role == "INJECTOR" and c.status == "OPEN"]
        shut = [c for c in by_date[d] if c.status == "SHUT"]
        if producers:
            chunks.append("WCONPROD")
            for c in producers:
                lr = 0.0 if c.liquid_rate is None else c.liquid_rate
                bhp = "1*" if c.bhp is None else f"{c.bhp:.3f}"
                chunks.append(f"  '{c.well}' 'OPEN' 'LRAT' 3* {lr:.3f} 1* {bhp} /")
            chunks.append("/")
        if injectors:
            chunks.append("WCONINJE")
            for c in injectors:
                rate = 0.0 if c.injection_rate is None else c.injection_rate
                bhp = "1*" if c.bhp is None else f"{c.bhp:.3f}"
                chunks.append(f"  '{c.well}' 'WATER' 'OPEN' 'RATE' {rate:.3f} 1* {bhp} /")
            chunks.append("/")
        if shut:
            chunks.append("WELOPEN")
            for c in shut:
                chunks.append(f"  '{c.well}' 'SHUT' /")
            chunks.append("/")
    return "\n".join(chunks) + ("\n" if chunks else "")
