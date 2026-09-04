from __future__ import annotations

import re
import shlex
from collections.abc import Callable
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


ScaleProvider = Callable[[date | None], tuple[dict[str, float], dict[str, float]]]


def _scale_schedule(
    text: str,
    *,
    scale_provider: ScaleProvider,
    max_wlpr: float,
    effective_from: date | None,
) -> str:
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
            producer_scale, injector_scale = scale_provider(current_date) if active else ({}, {})
            out.append(_rewrite_record(record, keyword, producer_scale, injector_scale, max_wlpr))
            record = ""
    if record:
        out.append(record)
    return "".join(out)


def scale_schedule_text(
    text: str,
    *,
    producer_scale: dict[str, float],
    injector_scale: dict[str, float],
    max_wlpr: float = 500.0,
    effective_from: date | None = None,
) -> str:
    """Scale WCON targets without changing pre-optimization history."""

    def constant_scale(_: date | None) -> tuple[dict[str, float], dict[str, float]]:
        return producer_scale, injector_scale

    return _scale_schedule(
        text,
        scale_provider=constant_scale,
        max_wlpr=max_wlpr,
        effective_from=effective_from,
    )


def _month_index(value: date) -> int:
    return value.year * 12 + value.month - 1


def _interpolate_monthly(value_date: date, node_dates: tuple[date, ...], values: tuple[float, ...]) -> float:
    if len(node_dates) != len(values) or not node_dates:
        raise ValueError("node_dates and policy values must have the same non-zero length")
    positions = tuple(_month_index(item) for item in node_dates)
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError("node_dates must be strictly increasing by month")
    position = _month_index(value_date)
    if position <= positions[0]:
        return float(values[0])
    if position >= positions[-1]:
        return float(values[-1])
    for idx, (left, right) in enumerate(zip(positions, positions[1:])):
        if left <= position <= right:
            weight = (position - left) / (right - left)
            return float(values[idx] + weight * (values[idx + 1] - values[idx]))
    raise RuntimeError("failed to bracket policy date")


def scale_schedule_with_policy(
    text: str,
    *,
    well_groups: dict[str, int],
    producer_group_nodes: dict[int, tuple[float, ...]],
    injector_group_nodes: dict[int, tuple[float, ...]],
    node_dates: tuple[date, ...],
    effective_from: date,
    max_wlpr: float = 500.0,
) -> str:
    """Apply smooth group-level policy nodes to native monthly WCON records.

    Scales are linearly interpolated in calendar months between the supplied
    policy nodes.  All records before ``effective_from`` are copied unchanged.
    """
    if not node_dates:
        raise ValueError("at least one policy node is required")
    for mapping in (producer_group_nodes, injector_group_nodes):
        for values in mapping.values():
            if len(values) != len(node_dates):
                raise ValueError("every group policy must provide one value per node date")
            if any(value <= 0 for value in values):
                raise ValueError("policy scales must be positive")

    def policy_for(current_date: date | None) -> tuple[dict[str, float], dict[str, float]]:
        if current_date is None:
            return {}, {}
        producers: dict[str, float] = {}
        injectors: dict[str, float] = {}
        for well, group in well_groups.items():
            if group in producer_group_nodes:
                producers[well] = _interpolate_monthly(current_date, node_dates, producer_group_nodes[group])
            if group in injector_group_nodes:
                injectors[well] = _interpolate_monthly(current_date, node_dates, injector_group_nodes[group])
        return producers, injectors

    return _scale_schedule(
        text,
        scale_provider=policy_for,
        max_wlpr=max_wlpr,
        effective_from=effective_from,
    )
