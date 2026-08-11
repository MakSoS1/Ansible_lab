import numpy as np
import pytest

from ecup_matching.ml.v6_teacher_gate import (
    GATE_COVERAGES,
    build_teacher_gated_scores,
    disagreement_gate_mask,
)


def _signals(n=40):
    x = np.linspace(0.01, 0.99, n)
    return {
        "weak": x,
        "sparse": np.roll(x, 3),
        "explicit": np.roll(x[::-1], 5),
        "contrastive": np.sin(np.linspace(0.0, 3.0, n)) * 0.4 + 0.5,
        "teacher": np.cos(np.linspace(0.0, 4.0, n)) * 2.0,
        "typed_explicit": np.roll(x, 7),
    }


def _categories(n=40):
    return np.array(["A"] * (n // 2) + ["B"] * (n - n // 2), dtype=object)


def test_gate_coverages_are_frozen_before_evaluation():
    assert GATE_COVERAGES == (0.25, 0.40, 0.55, 0.70, 0.85, 0.95)


def test_disagreement_gate_is_target_free_and_category_balanced():
    retained = {k: v for k, v in _signals().items() if k != "teacher"}
    mask = disagreement_gate_mask(retained, _categories(), coverage=0.40)
    assert mask.dtype == bool
    assert mask.sum() == 16
    assert mask[:20].sum() == 8
    assert mask[20:].sum() == 8


def test_gated_teacher_does_not_read_unselected_teacher_values():
    signals = _signals()
    categories = _categories()
    first, mask = build_teacher_gated_scores(signals, categories, coverage=0.40)
    changed = dict(signals)
    changed_teacher = np.array(signals["teacher"], copy=True)
    changed_teacher[~mask] += 100000.0
    changed["teacher"] = changed_teacher
    second, mask2 = build_teacher_gated_scores(changed, categories, coverage=0.40)
    np.testing.assert_array_equal(mask, mask2)
    np.testing.assert_allclose(first["teacher"], second["teacher"], atol=0.0, rtol=0.0)
    for name in ("weak", "sparse", "explicit", "contrastive", "typed_explicit"):
        np.testing.assert_allclose(first[name], second[name], atol=0.0, rtol=0.0)


def test_selected_teacher_values_can_change_selected_teacher_signal():
    signals = _signals()
    categories = _categories()
    first, mask = build_teacher_gated_scores(signals, categories, coverage=0.40)
    changed = dict(signals)
    selected = np.flatnonzero(mask)
    changed_teacher = np.array(signals["teacher"], copy=True)
    changed_teacher[selected] = changed_teacher[selected][::-1]
    changed["teacher"] = changed_teacher
    second, _ = build_teacher_gated_scores(changed, categories, coverage=0.40)
    assert not np.allclose(first["teacher"][mask], second["teacher"][mask])


def test_invalid_coverage_is_rejected():
    retained = {k: v for k, v in _signals().items() if k != "teacher"}
    with pytest.raises(ValueError, match="coverage"):
        disagreement_gate_mask(retained, _categories(), coverage=0.0)
    with pytest.raises(ValueError, match="coverage"):
        disagreement_gate_mask(retained, _categories(), coverage=1.1)
