"""Every first-party module the v6 entrypoint needs must ship in the archive.

The submission runs offline with no repository checkout, so a module that is
imported but not copied into the ZIP fails at inference time rather than at
build time — and a module left at its stale v5-base version fails silently by
producing the wrong predictions. Both cost a full submission attempt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ecup_matching.ci.runtime_closure import (
    EXTERNAL_RUNTIME_PACKAGES,
    REPO_ROOT,
    copy_runtime_closure,
    missing_from,
    runtime_import_closure,
    runtime_modules,
)


WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ecup-v6-final-submit.yml"


def test_closure_reaches_the_modules_the_runtime_actually_uses():
    closure = runtime_import_closure()
    for expected in (
        "ecup_matching/submission/run_v6.py",
        "ecup_matching/submission/predict_v6.py",
        "ecup_matching/submission/predict_v5.py",
        "ecup_matching/submission/v6_fast.py",
        "ecup_matching/submission/v6_parallel.py",
        "ecup_matching/ml/data_subset.py",
        "ecup_matching/ml/features.py",
        "ecup_matching/ml/features_v2.py",
        "ecup_matching/ml/textnorm.py",
        "ecup_matching/ml/v5_production.py",
        "ecup_matching/ml/v6_teacher_gate.py",
    ):
        assert expected in closure, f"{expected} is missing from the runtime closure"


def test_closure_includes_package_markers():
    closure = runtime_import_closure()
    assert "ecup_matching/__init__.py" in closure
    assert "ecup_matching/ml/__init__.py" in closure
    assert "ecup_matching/submission/__init__.py" in closure


def test_copying_the_closure_satisfies_the_verifier(tmp_path: Path):
    assert missing_from(tmp_path), "an empty tree must be reported as incomplete"
    copied = copy_runtime_closure(tmp_path)
    assert copied == runtime_import_closure()
    assert missing_from(tmp_path) == []


def test_every_closure_file_exists_in_the_repository_or_is_a_package_marker():
    from ecup_matching.ci.runtime_closure import PACKAGE_MARKERS

    for relative in runtime_import_closure():
        assert (REPO_ROOT / relative).is_file() or relative in PACKAGE_MARKERS, relative


def test_runtime_does_not_import_training_only_modules():
    """Training-only imports broke the offline organizer smoke before."""
    banned = {
        "ecup_matching.ml.v5_meta_blend",
        "ecup_matching.ml.split",
        "ecup_matching.ml.v5_validation",
        "ecup_matching.ml.v5_evaluation",
        "ecup_matching.ml.metrics",
        "ecup_matching.ml.v5_contrastive_data",
    }
    leaked = sorted(set(runtime_modules()) & banned)
    assert not leaked, f"runtime path pulls in training-only modules: {leaked}"


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="final-submit workflow not present")
def test_final_submit_workflow_packages_the_closure_programmatically():
    """A hand-maintained cp list is exactly how v6_parallel.py would have been lost."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "ecup_matching.ci.runtime_closure --copy-into" in text, (
        "the packaging step must copy the derived import closure, not a manual list"
    )
    assert "ecup_matching.ci.runtime_closure --verify" in text, (
        "the packaging step must verify the archive against the import closure"
    )


def test_only_legacy_ecup_is_resolved_from_outside_the_package():
    assert EXTERNAL_RUNTIME_PACKAGES == ("legacy_ecup",)
    for relative in runtime_import_closure():
        assert relative.startswith("ecup_matching/"), relative
