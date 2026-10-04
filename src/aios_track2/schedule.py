from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
MAX_WLPR = 500.0


class WellRole(StrEnum):
    PRODUCER = "producer"
    INJECTOR = "injector"


@dataclass(frozen=True)
class Control:
    date: date
    well: str
    status: str = "OPEN"
    role: WellRole = WellRole.PRODUCER
    wlpr: float = 0.0
    wwir: float = 0.0
    bhp: float | None = None
    thp: float | None = None
    convert: bool = False


@dataclass(frozen=True)
class Schedule:
    controls: tuple[Control, ...]

    def sorted(self) -> Schedule:
        return Schedule(controls=tuple(sorted(self.controls, key=lambda item: (item.date, item.well))))

    @property
    def sha256(self) -> str:
        return sha256(write_schedule_text(self).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    well: str | None = None
    date: date | None = None


@dataclass(frozen=True)
class ConstraintSet:
    max_wlpr_m3_day: float = MAX_WLPR
    known_wells: frozenset[str] | None = None
    min_bhp: float = 20.0
    max_bhp: float = 350.0
    max_field_liquid: float | None = None
    max_field_injection: float | None = None


@dataclass(frozen=True)
class ProjectionResult:
    schedule: Schedule
    accepted: bool
    violations: tuple[Violation, ...]
    projected: Schedule = field(default_factory=lambda: Schedule(controls=()))


def project_schedule(schedule: Schedule, constraints: ConstraintSet | None = None) -> ProjectionResult:
    constraints = constraints or ConstraintSet()
    ordered = schedule.sorted()
    violations: list[Violation] = []
    seen: set[tuple[date, str]] = set()
    roles: dict[str, WellRole] = {}
    projected: list[Control] = []

    for control in ordered.controls:
        key = (control.date, control.well)
        if key in seen:
            violations.append(Violation("DUPLICATE_CONTROL", "duplicate well/date control", control.well, control.date))
            continue
        seen.add(key)
        if constraints.known_wells is not None and control.well not in constraints.known_wells:
            violations.append(Violation("UNKNOWN_WELL", f"unknown well {control.well}", control.well, control.date))
            continue
        if control.wlpr < 0 or control.wwir < 0:
            violations.append(Violation("NEGATIVE_RATE", "negative rate", control.well, control.date))
            continue
        wlpr = min(control.wlpr, constraints.max_wlpr_m3_day)
        if control.wlpr > constraints.max_wlpr_m3_day:
            violations.append(
                Violation("WLPR_LIMIT", f"WLPR {control.wlpr} exceeds {constraints.max_wlpr_m3_day}", control.well, control.date)
            )
            continue
        if control.bhp is not None and not constraints.min_bhp <= control.bhp <= constraints.max_bhp:
            violations.append(Violation("PRESSURE_BOUNDS", "BHP outside bounds", control.well, control.date))
            continue
        previous_role = roles.get(control.well)
        role = WellRole.INJECTOR if control.convert or control.role == WellRole.INJECTOR or control.wwir > 0 else control.role
        if previous_role == WellRole.INJECTOR and role == WellRole.PRODUCER:
            violations.append(
                Violation("IRREVERSIBLE_CONVERSION", "injector cannot convert back to producer", control.well, control.date)
            )
            continue
        roles[control.well] = role
        projected.append(
            Control(
                date=control.date,
                well=control.well,
                status=control.status.upper(),
                role=role,
                wlpr=0.0 if role == WellRole.INJECTOR else wlpr,
                wwir=control.wwir if role == WellRole.INJECTOR else 0.0,
                bhp=control.bhp,
                thp=control.thp,
                convert=role == WellRole.INJECTOR and previous_role == WellRole.PRODUCER,
            )
        )

    if constraints.max_field_liquid is not None:
        by_date: dict[date, float] = {}
        for control in projected:
            if control.role == WellRole.PRODUCER and control.status == "OPEN":
                by_date[control.date] = by_date.get(control.date, 0.0) + control.wlpr
        for day, total in by_date.items():
            if total > constraints.max_field_liquid:
                violations.append(Violation("INFRASTRUCTURE_OVERFLOW", f"field liquid {total}", date=day))

    projected_schedule = Schedule(controls=tuple(projected))
    hard = tuple(violations)
    accepted = not any(item.code in {"WLPR_LIMIT", "UNKNOWN_WELL", "NEGATIVE_RATE", "IRREVERSIBLE_CONVERSION", "PRESSURE_BOUNDS", "DUPLICATE_CONTROL", "INFRASTRUCTURE_OVERFLOW"} for item in hard)
    return ProjectionResult(schedule=ordered, accepted=accepted, violations=hard, projected=projected_schedule)


def _format_date(value: date) -> str:
    return f"{value.day} {MONTHS[value.month - 1]} {value.year}"


def write_schedule_text(schedule: Schedule) -> str:
    ordered = schedule.sorted()
    chunks: list[str] = []
    current_date: date | None = None
    producers: list[Control] = []
    injectors: list[Control] = []
    status: list[Control] = []

    def flush() -> None:
        nonlocal producers, injectors, status
        if current_date is None:
            return
        chunks.append(f"DATES\n  {_format_date(current_date)} /\n/\n")
        if producers:
            chunks.append("WCONPROD\n")
            for control in producers:
                chunks.append(f"  '{control.well}' '{control.status}' 'LRAT' 1* {control.wlpr:.3f} /\n")
            chunks.append("/\n")
        if injectors:
            chunks.append("WCONINJE\n")
            for control in injectors:
                chunks.append(f"  '{control.well}' 'WATER' '{control.status}' 'RATE' {control.wwir:.3f} /\n")
            chunks.append("/\n")
        if status:
            chunks.append("WELOPEN\n")
            for control in status:
                chunks.append(f"  '{control.well}' '{control.status}' /\n")
            chunks.append("/\n")
        producers, injectors, status = [], [], []

    for control in ordered.controls:
        if current_date != control.date:
            flush()
            current_date = control.date
        if control.status == "SHUT":
            status.append(control)
        elif control.role == WellRole.INJECTOR or control.wwir > 0:
            injectors.append(control)
        else:
            producers.append(control)
    flush()
    return "".join(chunks)


def write_schedule_inc(schedule: Schedule, path: Path) -> str:
    text = write_schedule_text(schedule)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def parse_schedule_inc(text: str) -> Schedule:
    controls: list[Control] = []
    current_date: date | None = None
    mode: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("--"):
            continue
        if line == "/":
            mode = None
            continue
        if line == "DATES":
            mode = "DATES"
            continue
        if line == "WCONPROD":
            mode = "WCONPROD"
            continue
        if line == "WCONINJE":
            mode = "WCONINJE"
            continue
        if line == "WELOPEN":
            mode = "WELOPEN"
            continue
        if mode == "DATES":
            parts = line.replace("/", " ").split()
            current_date = date(int(parts[2]), MONTHS.index(parts[1]) + 1, int(parts[0]))
            continue
        if current_date is None:
            continue
        tokens = [item.strip("'") for item in line.replace("/", " ").split() if item not in {"/", ""}]
        if mode == "WCONPROD" and len(tokens) >= 5:
            rate = float(tokens[-1] if tokens[-1] != "1*" else tokens[-2] if tokens[-2] != "1*" else 0.0)
            if tokens[3] == "1*":
                rate = float(tokens[4])
            controls.append(Control(date=current_date, well=tokens[0], status=tokens[1], role=WellRole.PRODUCER, wlpr=rate))
        elif mode == "WCONINJE" and len(tokens) >= 5:
            rate = float(tokens[4]) if tokens[3] in {"RATE", "BHP"} else float(tokens[-1])
            if tokens[3] == "RATE":
                rate = float(tokens[4])
            controls.append(
                Control(date=current_date, well=tokens[0], status=tokens[2], role=WellRole.INJECTOR, wwir=rate)
            )
        elif mode == "WELOPEN" and len(tokens) >= 2:
            controls.append(Control(date=current_date, well=tokens[0], status=tokens[1], wlpr=0.0))
    return Schedule(controls=tuple(controls)).sorted()
