"""The v7 archive must contain its whole import graph and nothing training-only.

The submission runs offline. A missing module fails at inference; a training-only
module drags in the split/metric/validation graph, which is what broke the
organizer smoke in an earlier iteration.
"""

from __future__ import annotations

from pathlib import Path

from ecup_matching.ci.runtime_closure import (
    PACKAGE_MARKERS,
    REPO_ROOT,
    V7_ENTRYPOINTS,
    copy_runtime_closure,
    missing_from,
    runtime_import_closure,
    runtime_modules,
)


def test_v7_closure_contains_the_inference_path():
    closure = runtime_import_closure(V7_ENTRYPOINTS)
    for expected in (
        "ecup_matching/submission/run_v7.py",
        "ecup_matching/submission/predict_v7.py",
        "ecup_matching/ml/v7_runtime.py",
        "ecup_matching/ml/v7_item_text.py",
        "ecup_matching/ml/textnorm.py",
    ):
        assert expected in closure, f"{expected} missing from the v7 runtime closure"


def test_v7_closure_excludes_the_training_graph():
    """v7_neural imports train_v5_teacher_fold and v5_teacher2_objective.

    Inference must reach the shared code through v7_runtime instead, or the
    archive has to ship the entire training dependency chain.
    """
    banned = {
        "ecup_matching.ml.v7_neural",
        "ecup_matching.ml.train_v5_teacher_fold",
        "ecup_matching.ml.v5_teacher2_objective",
        "ecup_matching.ml.split",
        "ecup_matching.ml.metrics",
        "ecup_matching.ml.v5_evaluation",
        "ecup_matching.ml.v5_validation",
        "ecup_matching.ml.train_v1",
        "ecup_matching.ml.reranker_data",
        "ecup_matching.ml.v5_contrastive_data",
        "ecup_matching.ml.features",
    }
    leaked = sorted(set(runtime_modules(V7_ENTRYPOINTS)) & banned)
    assert not leaked, f"v7 inference reaches training-only modules: {leaked}"


def test_v7_closure_stays_small():
    """A ballooning closure means inference re-acquired a training dependency."""
    closure = runtime_import_closure(V7_ENTRYPOINTS)
    assert len(closure) <= 12, (
        f"v7 runtime closure grew to {len(closure)} modules: {closure}"
    )


def test_copying_the_v7_closure_satisfies_the_verifier(tmp_path: Path):
    assert missing_from(tmp_path, V7_ENTRYPOINTS), "empty tree must report as incomplete"
    copied = copy_runtime_closure(tmp_path, V7_ENTRYPOINTS)
    assert copied == runtime_import_closure(V7_ENTRYPOINTS)
    assert missing_from(tmp_path, V7_ENTRYPOINTS) == []
    for marker in PACKAGE_MARKERS:
        assert (tmp_path / marker).is_file()


def test_v7_and_v6_closures_are_independent():
    """Generalizing the closure helper must not have changed the v6 archive."""
    v6 = runtime_import_closure()
    assert "ecup_matching/submission/run_v6.py" in v6
    assert "ecup_matching/submission/run_v7.py" not in v6
    v7 = runtime_import_closure(V7_ENTRYPOINTS)
    assert "ecup_matching/submission/run_v6.py" not in v7


def test_every_v7_closure_file_exists():
    for relative in runtime_import_closure(V7_ENTRYPOINTS):
        assert (REPO_ROOT / relative).is_file() or relative in PACKAGE_MARKERS, relative
