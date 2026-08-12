"""The production driver must save weights and must not report a quality number."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "ml" / "run_v7_production.py"
PROBE = Path(__file__).resolve().parents[1] / "ml" / "run_v7_fold0_probe.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _calls(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


def test_production_driver_persists_the_model():
    """The fold-0 probe discarded its weights; that is why a retrain was needed."""
    assert "save_pretrained" in _calls(SOURCE)
    assert "save_pretrained" not in _calls(PROBE), (
        "probe now saves weights: update the comment explaining why a refit was required"
    )


def test_production_driver_refuses_to_report_validation_quality():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"validation_metric_reported": False' in source
    for banned in ("macro_ap_report", "average_precision_score", "fold0_macro_average_precision"):
        assert banned not in source, (
            f"a production refit must not compute {banned}; it saw every development row"
        )


def test_production_driver_trains_on_every_development_row():
    source = SOURCE.read_text(encoding="utf-8")
    assert "285_210" in source
    assert "held_fold" not in source, "a production refit must not hold out a fold"


def test_production_driver_requires_cuda_and_the_frozen_split():
    source = SOURCE.read_text(encoding="utf-8")
    assert "requires CUDA" in source
    assert "IMMUTABLE_SPLIT_SHA" in source
    assert "leaked a human item" in source


def test_production_driver_module_is_importable_without_cuda():
    module = pytest.importorskip("ecup_matching.ml.run_v7_production")
    assert hasattr(module, "train_v7_production")
    assert hasattr(module, "main")
