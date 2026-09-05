from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from datetime import date

from .eclipse_schedule import _numeric, _parse_eclipse_date, _token_for_item


def cyclic_factor(
    current: date,
    *,
    high: float,
    low: float,
    period_months: int,
    origin: date,
) -> float:
    if period_months <= 0:
        raise ValueError("period_months must be positive")
    months = (current.year - origin.year) * 12 + (current.month - origin.month)
    cycle = (months // period_months) % 2
    return float(high if cycle == 0 else low)


@dataclass(frozen=True, slots=True)
class WellActionPlan:
    """Explicit well-level actions allowed by Track 2 on top of group rate policies.

    These operators rewrite native Eclipse schedule text after the economic start.
    They are not implied by the 18-D group-multiplier tournament vector.
    """

    shut_wells: tuple[str, ...] = ()
    convert_to_injector: dict[str, float] = field(default_factory=dict)
    cyclic_injectors: tuple[str, ...] = ()
    cyclic_high: float = 1.15
    cyclic_low: float = 0.85
    cyclic_period_months: int = 6
    cyclic_origin: date = date(2007, 1, 1)


def _tokens(record: str) -> list[str]:
    stripped = re.sub(r"--.*$", "", record, flags=re.M).strip()
    if not stripped.endswith("/"):
        return []
    body = stripped[:-1].strip()
    if not body:
        return []
    lexer = shlex.shlex(body, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _join(tokens: list[str], indent: str) -> str:
    return indent + " ".join(tokens) + " /\n"


def _wconprod_target_rate(tokens: list[str]) -> float | None:
    if len(tokens) < 3:
        return None
    mode = tokens[2].strip("'\"").upper()
    target_item = {"ORAT": 3, "WRAT": 4, "GRAT": 5, "LRAT": 6, "RESV": 7}.get(mode)
    if target_item is None:
        return None
    idx = _token_for_item(tokens, target_item)
    if idx is None:
        return None
    return _numeric(tokens[idx])


def _set_status(tokens: list[str], *, keyword: str, status: str) -> list[str]:
    out = list(tokens)
    token = f"'{status}'"
    if keyword == "WCONPROD" and len(out) >= 2:
        out[1] = token
    elif keyword == "WCONINJE" and len(out) >= 3:
        out[2] = token
    return out


def _scale_injector_rate(tokens: list[str], factor: float) -> list[str]:
    out = list(tokens)
    if len(out) < 4:
        return out
    mode = out[3].strip("'\"").upper()
    target_item = 4 if mode == "RATE" else 5 if mode == "RESV" else None
    if target_item is None:
        return out
    idx = _token_for_item(out, target_item)
    if idx is None:
        return out
    value = _numeric(out[idx])
    if value is None:
        return out
    out[idx] = f"{value * float(factor):.6f}"
    return out


def apply_well_actions(text: str, plan: WellActionPlan, *, effective_from: date) -> str:
    shut = set(plan.shut_wells)
    convert = dict(plan.convert_to_injector)
    cyclic = set(plan.cyclic_injectors)
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    keyword: str | None = None
    record = ""
    in_dates = False
    date_record = ""
    current_date: date | None = None
    pending_inje: list[str] = []

    def active() -> bool:
        return current_date is not None and current_date >= effective_from

    def flush_pending() -> None:
        nonlocal pending_inje
        if not pending_inje:
            return
        out.append("WCONINJE\n")
        out.extend(pending_inje)
        out.append("/\n")
        pending_inje = []

    def finish_record() -> None:
        nonlocal record
        indent = re.match(r"\s*", record).group(0)
        tokens = _tokens(record)
        well = tokens[0].strip("'\"") if tokens else ""
        if not tokens or keyword is None or not active():
            out.append(record)
            record = ""
            return
        if keyword == "WCONPROD" and well in convert:
            shut_tokens = _set_status(tokens, keyword="WCONPROD", status="SHUT")
            out.append(_join(shut_tokens, indent))
            rate = float(convert[well])
            pending_inje.append(f"{indent}'{well}' 'WATER' 'OPEN' 'RATE' {rate:.6f} 1* 300 /\n")
            record = ""
            return
        rewritten = tokens
        if well in shut:
            rewritten = _set_status(rewritten, keyword=keyword, status="SHUT")
        if keyword == "WCONINJE" and well in cyclic:
            factor = cyclic_factor(
                current_date or effective_from,
                high=plan.cyclic_high,
                low=plan.cyclic_low,
                period_months=plan.cyclic_period_months,
                origin=plan.cyclic_origin,
            )
            rewritten = _scale_injector_rate(rewritten, factor)
        out.append(_join(rewritten, indent) if rewritten != tokens else record)
        record = ""

    for line in lines:
        bare = re.sub(r"--.*$", "", line).strip().upper()
        if keyword is None and not in_dates and bare == "DATES":
            flush_pending()
            in_dates = True
            date_record = ""
            out.append(line)
            continue
        if in_dates:
            if not date_record and re.fullmatch(r"\s*/\s*(?:--.*)?\n?", line):
                out.append(line)
                in_dates = False
                continue
            date_record += line
            out.append(line)
            if "/" in re.sub(r"--.*$", "", line):
                parsed = _parse_eclipse_date(date_record)
                if parsed is not None:
                    current_date = parsed
                date_record = ""
            continue
        if keyword is None and bare in {"WCONPROD", "WCONINJE"}:
            keyword = bare
            out.append(line)
            continue
        if keyword is None:
            out.append(line)
            continue
        if not record and re.fullmatch(r"\s*/\s*(?:--.*)?\n?", line):
            out.append(line)
            if keyword == "WCONPROD":
                flush_pending()
            keyword = None
            continue
        record += line
        if "/" in re.sub(r"--.*$", "", line):
            finish_record()
    if record:
        out.append(record)
    flush_pending()
    return "".join(out)
