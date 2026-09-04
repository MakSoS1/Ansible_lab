from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np


def _fortran_records(path: Path) -> Iterator[bytes]:
    data = path.read_bytes()
    position = 0
    while position < len(data):
        if position + 4 > len(data):
            raise ValueError(f"truncated Fortran record marker in {path}")
        size = struct.unpack(">i", data[position : position + 4])[0]
        position += 4
        if size < 0 or position + size + 4 > len(data):
            raise ValueError(f"invalid Fortran record size {size} in {path}")
        payload = data[position : position + size]
        position += size
        closing = struct.unpack(">i", data[position : position + 4])[0]
        position += 4
        if closing != size:
            raise ValueError(f"Fortran record marker mismatch in {path}: {size} != {closing}")
        yield payload


def _decode_values(kind: str, payload: bytes) -> list[object]:
    if kind == "INTE" or kind == "LOGI":
        if len(payload) % 4:
            raise ValueError(f"invalid {kind} payload width")
        return np.frombuffer(payload, dtype=">i4").astype(np.int64).tolist()
    if kind == "REAL":
        if len(payload) % 4:
            raise ValueError("invalid REAL payload width")
        return np.frombuffer(payload, dtype=">f4").astype(np.float64).tolist()
    if kind == "DOUB":
        if len(payload) % 8:
            raise ValueError("invalid DOUB payload width")
        return np.frombuffer(payload, dtype=">f8").astype(np.float64).tolist()
    if kind == "CHAR":
        width = 8
    elif kind.startswith("C0") and kind[2:].isdigit():
        width = int(kind[2:])
    else:
        raise ValueError(f"unsupported Eclipse value type {kind!r}")
    if len(payload) % width:
        raise ValueError(f"invalid {kind} character payload width")
    return [payload[offset : offset + width].decode("ascii", errors="strict").strip() for offset in range(0, len(payload), width)]


def _eclipse_records(path: Path) -> Iterator[tuple[str, str, list[object]]]:
    records = iter(_fortran_records(path))
    while True:
        try:
            header = next(records)
        except StopIteration:
            return
        if len(header) != 16:
            raise ValueError(f"expected 16-byte Eclipse keyword header in {path}, got {len(header)}")
        keyword = header[:8].decode("ascii", errors="strict").strip()
        count = struct.unpack(">i", header[8:12])[0]
        kind = header[12:16].decode("ascii", errors="strict")
        if count < 0:
            raise ValueError(f"negative value count for {keyword} in {path}")
        if kind == "MESS":
            yield keyword, kind, []
            continue
        values: list[object] = []
        while len(values) < count:
            try:
                payload = next(records)
            except StopIteration as exc:
                raise ValueError(f"truncated data for Eclipse keyword {keyword} in {path}") from exc
            values.extend(_decode_values(kind, payload))
        if len(values) != count:
            raise ValueError(f"decoded value count mismatch for {keyword}: expected {count}, got {len(values)}")
        yield keyword, kind, values


def _single_case_file(root: Path, suffix: str) -> Path:
    candidates = sorted(root.rglob(f"*{suffix}"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one {suffix} beneath {root}, found {len(candidates)}")
    return candidates[0]


@dataclass(frozen=True, slots=True)
class EclSummary:
    start_date: date
    dates: tuple[date, ...]
    keywords: tuple[str, ...]
    wgnames: tuple[str, ...]
    units: tuple[str, ...]
    params: np.ndarray

    def _index(self, keyword: str, wgname: str | None) -> int:
        matches = [
            index
            for index, (candidate, group) in enumerate(zip(self.keywords, self.wgnames, strict=True))
            if candidate == keyword and (wgname is None or group == wgname)
        ]
        if not matches:
            qualifier = f":{wgname}" if wgname is not None else ""
            raise KeyError(f"summary vector {keyword}{qualifier} not found")
        if len(matches) > 1 and wgname is None:
            raise KeyError(f"summary vector {keyword} is ambiguous; provide wgname")
        return matches[0]

    def vector(self, keyword: str, wgname: str | None = None) -> np.ndarray:
        return self.params[:, self._index(keyword, wgname)].copy()

    def unit(self, keyword: str, wgname: str | None = None) -> str:
        return self.units[self._index(keyword, wgname)]


def load_summary_case(root: Path) -> EclSummary:
    smspec = _single_case_file(root, ".SMSPEC")
    unsmry = _single_case_file(root, ".UNSMRY")

    metadata: dict[str, list[object]] = {}
    for keyword, _kind, values in _eclipse_records(smspec):
        if keyword in {"KEYWORDS", "WGNAMES", "UNITS", "STARTDAT"} and keyword not in metadata:
            metadata[keyword] = values
    required = {"KEYWORDS", "WGNAMES", "UNITS", "STARTDAT"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"SMSPEC missing required records: {missing}")

    keywords = tuple(str(value) for value in metadata["KEYWORDS"])
    wgnames = tuple(str(value) for value in metadata["WGNAMES"])
    units = tuple(str(value) for value in metadata["UNITS"])
    if not (len(keywords) == len(wgnames) == len(units)):
        raise ValueError("SMSPEC keyword/name/unit widths differ")
    start_values = metadata["STARTDAT"]
    if len(start_values) < 3:
        raise ValueError("STARTDAT must contain day, month and year")
    start_date = date(int(start_values[2]), int(start_values[1]), int(start_values[0]))

    parameter_rows = [values for keyword, _kind, values in _eclipse_records(unsmry) if keyword == "PARAMS"]
    if not parameter_rows:
        raise ValueError("UNSMRY contains no PARAMS records")
    widths = {len(row) for row in parameter_rows}
    if widths != {len(keywords)}:
        raise ValueError(f"PARAMS width mismatch: expected {len(keywords)}, observed {sorted(widths)}")
    params = np.asarray(parameter_rows, dtype=np.float64)

    try:
        time_index = next(index for index, keyword in enumerate(keywords) if keyword == "TIME")
    except StopIteration as exc:
        raise ValueError("SMSPEC contains no TIME vector") from exc
    dates = tuple((start_date + timedelta(days=float(days))) for days in params[:, time_index])
    return EclSummary(start_date, dates, keywords, wgnames, units, params)
