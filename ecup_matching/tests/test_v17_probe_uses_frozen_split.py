"""D043 cost a dispatched GPU run; this stops the v17 driver repeating it.

`_build_immutable_manifest` recomputes the split and produces
`d1b31023...` instead of the immutable `aae58fb4...`, so a driver that calls it
dies on dispatch — which is what run 32168884723 did. The frozen loader is not
optional, and a source-level check is the only thing that catches a regression
without spending another GPU slot to find out.
"""

from __future__ import annotations

import ast
from pathlib import Path


DRIVER = Path(__file__).resolve().parents[1] / "ml" / "run_v17_weakscale_probe.py"


def _imported_names() -> set[str]:
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def test_driver_imports_the_frozen_manifest_loader():
    assert "_load_immutable_manifest" in _imported_names()


def test_driver_never_imports_the_recomputing_builder():
    assert "_build_immutable_manifest" not in _imported_names()


def test_driver_calls_only_the_frozen_loader():
    source = DRIVER.read_text(encoding="utf-8")
    assert "_load_immutable_manifest(" in source
    # The name may appear in the explanatory comment, but never as a call.
    assert "_build_immutable_manifest(" not in source
