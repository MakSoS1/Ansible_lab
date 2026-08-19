import numpy as np
import pytest

from ecup_matching.ml.v6_fast_ablation import (
    CANDIDATE_SPECS,
    build_fast_candidate_scores,
)


def _signals(n=8):
    base = np.linspace(0.05, 0.95, n)
    return {
        "weak": base,
        "sparse": base[::-1],
        "explicit": np.roll(base, 1),
        "contrastive": np.roll(base, 2),
        "teacher": np.roll(base, 3),
        "typed_explicit": np.roll(base, 4),
    }


def test_candidate_specs_are_frozen_and_runtime_ordered():
    assert tuple(CANDIDATE_SPECS) == (
        "structured_only",
        "no_teacher",
        "no_contrastive",
    )
    assert CANDIDATE_SPECS["structured_only"].required_expensive_signals == ()
    assert CANDIDATE_SPECS["no_teacher"].required_expensive_signals == ("contrastive",)
    assert CANDIDATE_SPECS["no_contrastive"].required_expensive_signals == ("teacher",)


def test_no_teacher_candidate_never_reads_teacher_values():
    signals = _signals()
    first = build_fast_candidate_scores(signals, "no_teacher")
    modified = dict(signals)
    modified["teacher"] = np.linspace(100.0, 200.0, len(signals["teacher"]))
    second = build_fast_candidate_scores(modified, "no_teacher")
    for name in first:
        np.testing.assert_allclose(first[name], second[name])


def test_structured_only_never_reads_either_neural_signal():
    signals = _signals()
    first = build_fast_candidate_scores(signals, "structured_only")
    modified = dict(signals)
    modified["teacher"] = np.full(8, -1000.0)
    modified["contrastive"] = np.full(8, 1000.0)
    second = build_fast_candidate_scores(modified, "structured_only")
    for name in first:
        np.testing.assert_allclose(first[name], second[name])


def test_candidate_keeps_six_signal_interface_and_finite_values():
    for candidate in CANDIDATE_SPECS:
        scores = build_fast_candidate_scores(_signals(), candidate)
        assert tuple(scores) == (
            "weak",
            "sparse",
            "explicit",
            "contrastive",
            "teacher",
            "typed_explicit",
        )
        assert all(np.isfinite(v).all() and len(v) == 8 for v in scores.values())


def test_unknown_candidate_is_rejected():
    with pytest.raises(ValueError, match="unknown fast candidate"):
        build_fast_candidate_scores(_signals(), "magic")
