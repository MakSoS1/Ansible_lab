from __future__ import annotations

import re
import shlex
from datetime import date, datetime


def _width(token: str) -> int:
    m = re.fullmatch(r"(\d+)\*", token)
    return int(m.group(1)) if m else 1


def _token_for_item(tokens: list[str], item: int) -> int | None:
    logical = 0
    for idx, token in enumerate(tokens):
        width = _width(token)
        if logical <= item < logical + width:
            if width > 1:
                return None
            return idx
        logical += width
    return None


def _numeric(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _parse_eclipse_date(record: str) -> date | None:
    clean = re.sub(r"--.*$", "", record, flags=re.M).strip().rstrip("/").strip()
    clean = clean.replace("'", "").replace('"', "")
    for pattern in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(clean, pattern).date()
        except ValueError:
            pass
    return None


def _rewrite_record(
    record: str,
    keyword: str,
    producer_scale: dict[str, float],
    injector_scale: dict[str, float],
    max_wlpr: float,
) -> str:
    stripped = re.sub(r"--.*$", "", record, flags=re.M).strip()
    if not stripped.endswith("/"):
        return record
    body = stripped[:-1].strip()
    if not body:
        return record
    lexer = shlex.shlex(body, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens = list(lexer)
    if not tokens:
        return record
    well = tokens[0].strip("'\"")
    if keyword == "WCONPROD" and well in producer_scale and len(tokens) >= 3:
        mode = tokens[2].strip("'\"").upper()
        target_item = {"ORAT": 3, "WRAT": 4, "GRAT": 5, "LRAT": 6, "RESV": 7}.get(mode)
        if target_item is not None:
            idx = _token_for_item(tokens, target_item)
            if idx is not None:
                value = _numeric(tokens[idx])
                if value is not None:
                    scaled = value * float(producer_scale[well])
                    if mode == "LRAT":
                        scaled = min(scaled, float(max_wlpr))
                    tokens[idx] = f"{scaled:.6f}"
    elif keyword == "WCONINJE" and well in injector_scale and len(tokens) >= 4:
        mode = tokens[3].strip("'\"").upper()
        target_item = 4 if mode == "RATE" else 5 if mode == "RESV" else None
        if target_item is not None:
            idx = _token_for_item(tokens, target_item)
            if idx is not None:
                value = _numeric(tokens[idx])
                if value is not None:
                    tokens[idx] = f"{value * float(injector_scale[well]):.6f}"
    indent = re.match(r"\s*", record).group(0)
    return indent + " ".join(tokens) + " /\n"


def scale_schedule_text(
    text: str,
    *,
    producer_scale: dict[str, float],
    injector_scale: dict[str, float],
    max_wlpr: float = 500.0,
    effective_from: date | None = None,
) -> str:
    """Scale WCON targets without changing pre-optimization history.

    When ``effective_from`` is supplied, records are modified only after a DATES record reaches
    that date. WCON records before the first known date are left untouched. This preserves the
    historical reservoir state while still allowing post-2007 DoE perturbations.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    keyword: str | None = None
    record = ""
    in_dates = False
    date_record = ""
    current_date: date | None = None

    for line in lines:
        bare = re.sub(r"--.*$", "", line).strip().upper()

        if keyword is None and not in_dates and bare == "DATES":
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
            keyword = None
            continue
        record += line
        if "/" in re.sub(r"--.*$", "", line):
            active = effective_from is None or (current_date is not None and current_date >= effective_from)
            out.append(
                _rewrite_record(
                    record,
                    keyword,
                    producer_scale if active else {},
                    injector_scale if active else {},
                    max_wlpr,
                )
            )
            record = ""
    if record:
        out.append(record)
    return "".join(out)
