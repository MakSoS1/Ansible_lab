"""Every long-running phase must report progress. This is the enforcement.

A v7 production run spent minutes in a single-threaded phase emitting nothing,
which was indistinguishable from a hung job or an idle GPU. The rule is now a
test, not a habit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ML = Path(__file__).resolve().parents[1] / "ml"

# Modules that iterate over competition-scale collections. Each must emit a
# heartbeat that is bounded in wall-clock time, not only in units completed.
MUST_REPORT_PROGRESS = (
    "run_v7_outer_oof.py",
    "v7_runtime.py",
)


@pytest.mark.parametrize("name", MUST_REPORT_PROGRESS)
def test_long_running_module_uses_the_progress_reporter(name):
    source = (ML / name).read_text(encoding="utf-8")
    assert "ProgressReporter" in source, (
        f"{name} loops over competition-scale data and must report progress; "
        "import ProgressReporter from ecup_matching.ml.progress"
    )


@pytest.mark.parametrize("name", MUST_REPORT_PROGRESS)
def test_progress_reporters_are_time_bounded(name):
    """every_seconds must be set, or a slow phase can still go silent."""
    tree = ast.parse((ML / name).read_text(encoding="utf-8"), filename=name)
    constructions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ProgressReporter"
    ]
    assert constructions, f"{name} references ProgressReporter but never builds one"
    for call in constructions:
        keywords = {kw.arg for kw in call.keywords}
        assert "every_seconds" in keywords, (
            f"{name}:{call.lineno} builds a ProgressReporter without every_seconds; "
            "a unit-only interval cannot bound silence when each unit is slow"
        )


def test_the_phase_that_hid_the_stall_no_longer_recomputes_the_split():
    """The frozen manifest replaces a multi-minute silent feature rebuild."""
    frozen = (ML / "run_v7_production_frozen.py").read_text(encoding="utf-8")
    assert "_load_immutable_manifest" in frozen
    assert "_build_immutable_manifest" in frozen


def test_every_v7_gpu_driver_runs_from_the_frozen_split():
    """A driver that recomputes the split trains against a different fold map."""
    for driver in ("run_v7_production", "run_v7_outer_oof", "run_v7_fold0_probe"):
        frozen = ML / f"{driver}_frozen.py"
        assert frozen.is_file(), (
            f"{driver} has no frozen wrapper; dispatching it directly would "
            "recompute the split and drift from the immutable SHA"
        )
        assert "_load_immutable_manifest" in frozen.read_text(encoding="utf-8")
