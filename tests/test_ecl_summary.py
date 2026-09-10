from __future__ import annotations

import struct
from datetime import date
from pathlib import Path

import numpy as np

from aios_track2.ecl_summary import load_summary_case


def _fortran(payload: bytes) -> bytes:
    marker = struct.pack(">i", len(payload))
    return marker + payload + marker


def _keyword(name: str, kind: str, values: list[object]) -> bytes:
    header = name.ljust(8).encode("ascii") + struct.pack(">i", len(values)) + kind.encode("ascii")
    if kind == "CHAR":
        payload = b"".join(str(value).ljust(8)[:8].encode("ascii") for value in values)
    elif kind == "INTE":
        payload = struct.pack(f">{len(values)}i", *(int(value) for value in values))
    elif kind == "REAL":
        payload = struct.pack(f">{len(values)}f", *(float(value) for value in values))
    else:
        raise AssertionError(kind)
    return _fortran(header) + _fortran(payload)


def test_load_summary_case_reads_dates_and_named_vectors(tmp_path: Path) -> None:
    smspec = b"".join(
        (
            _keyword("KEYWORDS", "CHAR", ["TIME", "FOPT", "WOPT"]),
            _keyword("WGNAMES", "CHAR", [":+:+:+:+", ":+:+:+:+", "P1"]),
            _keyword("UNITS", "CHAR", ["DAYS", "SM3", "SM3"]),
            _keyword("STARTDAT", "INTE", [1, 1, 2007, 0, 0, 0]),
        )
    )
    unsmry = b"".join(
        (
            _keyword("PARAMS", "REAL", [0.0, 100.0, 10.0]),
            _keyword("PARAMS", "REAL", [31.0, 130.0, 13.0]),
        )
    )
    (tmp_path / "CASE.SMSPEC").write_bytes(smspec)
    (tmp_path / "CASE.UNSMRY").write_bytes(unsmry)

    summary = load_summary_case(tmp_path)

    assert summary.start_date == date(2007, 1, 1)
    assert summary.dates == (date(2007, 1, 1), date(2007, 2, 1))
    np.testing.assert_allclose(summary.vector("FOPT"), [100.0, 130.0])
    np.testing.assert_allclose(summary.vector("WOPT", "P1"), [10.0, 13.0])
    assert summary.unit("WOPT", "P1") == "SM3"


def test_load_summary_case_rejects_parameter_width_mismatch(tmp_path: Path) -> None:
    (tmp_path / "CASE.SMSPEC").write_bytes(
        b"".join(
            (
                _keyword("KEYWORDS", "CHAR", ["TIME", "FOPT"]),
                _keyword("WGNAMES", "CHAR", [":+:+:+:+", ":+:+:+:+"]),
                _keyword("UNITS", "CHAR", ["DAYS", "SM3"]),
                _keyword("STARTDAT", "INTE", [1, 1, 2007, 0, 0, 0]),
            )
        )
    )
    (tmp_path / "CASE.UNSMRY").write_bytes(_keyword("PARAMS", "REAL", [0.0]))

    try:
        load_summary_case(tmp_path)
    except ValueError as exc:
        assert "width" in str(exc).lower()
    else:
        raise AssertionError("expected parameter width mismatch to fail closed")
