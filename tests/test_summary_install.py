from __future__ import annotations

import hashlib
from pathlib import Path

from aios_track2.summary_install import install_training_summary
from aios_track2.summary_requests import build_training_summary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_install_training_summary_changes_only_summary_include(tmp_path: Path) -> None:
    deck = tmp_path / "MODEL_Z.data"
    grid = tmp_path / "Model_Z_grid.inc"
    schedule = tmp_path / "Model_Z_sch.inc"
    summary = tmp_path / "Model_Z_summary.inc"
    deck.write_text(
        "RUNSPEC\n"
        "INCLUDE\n"
        " 'Model_Z_grid.inc' /\n"
        "SUMMARY\n"
        "INCLUDE\n"
        " 'Model_Z_summary.inc' /\n"
        "SCHEDULE\n"
        "INCLUDE\n"
        " 'Model_Z_sch.inc' /\n"
        "END\n",
        encoding="utf-8",
    )
    grid.write_text("DIMENS\n 2 2 1 /\n", encoding="utf-8")
    schedule.write_text("DATES\n 1 JAN 2007 /\n/\n", encoding="utf-8")
    summary.write_text("TIME\n", encoding="utf-8")

    immutable_before = {path.name: _sha(path) for path in (deck, grid, schedule)}
    old_summary_hash = _sha(summary)
    report = install_training_summary(tmp_path)

    assert {path.name: _sha(path) for path in (deck, grid, schedule)} == immutable_before
    assert _sha(summary) != old_summary_hash
    assert report.summary_path == "Model_Z_summary.inc"
    assert report.before_sha256 == old_summary_hash
    assert report.after_sha256 == _sha(summary)
    assert report.changed_files == ("Model_Z_summary.inc",)
    assert set(report.unchanged_files) == {"MODEL_Z.data", "Model_Z_grid.inc", "Model_Z_sch.inc"}
    assert "WOPR\n/\n" in summary.read_text(encoding="utf-8")


def test_install_training_summary_requires_exactly_one_summary_include(tmp_path: Path) -> None:
    (tmp_path / "MODEL_Z.data").write_text("SUMMARY\n", encoding="utf-8")
    try:
        install_training_summary(tmp_path)
    except ValueError as exc:
        assert "exactly one" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for missing summary include")


def test_training_summary_leaves_time_axes_to_opm_output() -> None:
    summary = build_training_summary()
    assert "TIME" not in summary
    assert "YEARS" not in summary
    assert "\nFOPR\n" in summary
