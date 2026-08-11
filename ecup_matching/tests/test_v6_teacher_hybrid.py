import numpy as np

from ecup_matching.ml.v6_teacher_hybrid import (
    HYBRID_COVERAGES,
    build_crossfit_hybrid_teacher,
)


def _signals(n=100):
    x = np.linspace(0.01, 0.99, n)
    return {
        "weak": x,
        "sparse": np.roll(x, 5),
        "explicit": np.roll(x[::-1], 9),
        "contrastive": np.sin(np.linspace(0.0, 7.0, n)) * 0.4 + 0.5,
        "typed_explicit": np.roll(x, 13),
    }


def test_hybrid_coverages_are_frozen_before_evaluation():
    assert HYBRID_COVERAGES == (0.25, 0.40, 0.55, 0.70, 0.85)


def test_unselected_held_teacher_values_do_not_affect_hybrid_score():
    n = 100
    folds = np.repeat(np.arange(5), n // 5)
    categories = np.array([f"C{i % 4}" for i in range(n)], dtype=object)
    teacher = np.sin(np.linspace(0.0, 9.0, n)) * 2.0
    first = build_crossfit_hybrid_teacher(
        _signals(n), teacher, categories, folds, coverage=0.40
    )
    changed = teacher.copy()
    changed[~first["teacher_selected"]] += 10000.0
    second = build_crossfit_hybrid_teacher(
        _signals(n), changed, categories, folds, coverage=0.40
    )
    np.testing.assert_array_equal(first["teacher_selected"], second["teacher_selected"])
    np.testing.assert_allclose(
        first["hybrid_teacher_score"],
        second["hybrid_teacher_score"],
        atol=0.0,
        rtol=0.0,
    )


def test_hybrid_scores_all_rows_and_uses_exact_requested_gate_fraction_per_category():
    n = 100
    folds = np.repeat(np.arange(5), n // 5)
    categories = np.array(["A"] * 50 + ["B"] * 50, dtype=object)
    teacher = np.cos(np.linspace(0.0, 5.0, n))
    result = build_crossfit_hybrid_teacher(
        _signals(n), teacher, categories, folds, coverage=0.40
    )
    score = result["hybrid_teacher_score"]
    mask = result["teacher_selected"]
    assert score.shape == (n,)
    assert np.isfinite(score).all()
    assert ((score >= 0.0) & (score <= 1.0)).all()
    assert mask[:50].sum() == 20
    assert mask[50:].sum() == 20


def test_held_selected_teacher_uses_outer_train_calibration_only():
    n = 100
    folds = np.repeat(np.arange(5), n // 5)
    categories = np.array([f"C{i % 5}" for i in range(n)], dtype=object)
    teacher = np.linspace(-3.0, 3.0, n)
    first = build_crossfit_hybrid_teacher(
        _signals(n), teacher, categories, folds, coverage=0.70
    )
    held = folds == 2
    selected_held = held & first["teacher_selected"]
    assert selected_held.any()
    changed = teacher.copy()
    # Changing unselected values inside the same held fold must not change
    # calibration of selected held rows, because calibration comes from outer train.
    changed[held & ~first["teacher_selected"]] -= 5000.0
    second = build_crossfit_hybrid_teacher(
        _signals(n), changed, categories, folds, coverage=0.70
    )
    np.testing.assert_allclose(
        first["hybrid_teacher_score"][selected_held],
        second["hybrid_teacher_score"][selected_held],
        atol=0.0,
        rtol=0.0,
    )
