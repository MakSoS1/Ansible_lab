from __future__ import annotations

import re
import shlex


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


def _rewrite_record(record: str, keyword: str, producer_scale: dict[str, float], injector_scale: dict[str, float], max_wlpr: float) -> str:
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
) -> str:
    """Scale active WCONPROD/WCONINJE targets while preserving all other schedule logic.

    The routine deliberately modifies only records whose control mode has an explicit numeric
    active target. Defaults, pressure limits, dates, group controls and unrelated wells are left
    unchanged. This makes DoE perturbations much safer than regenerating the entire history.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    keyword: str | None = None
    record = ""
    for line in lines:
        bare = re.sub(r"--.*$", "", line).strip().upper()
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
            out.append(_rewrite_record(record, keyword, producer_scale, injector_scale, max_wlpr))
            record = ""
    if record:
        out.append(record)
    return "".join(out)
