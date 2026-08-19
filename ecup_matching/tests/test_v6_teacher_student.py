import numpy as np

from ecup_matching.ml.v6_teacher_student import (
    DEFAULT_TEACHER_STUDENT_PARAMS,
    crossfit_teacher_student,
)


def _scores(n=120):
    x = np.linspace(0.01, 0.99, n)
    return {
        "weak": x,
        "sparse": np.roll(x, 7),
        "explicit": np.roll(x[::-1], 11),
        "contrastive": np.sin(np.linspace(0.0, 7.0, n)) * 0.4 + 0.5,
        "typed_explicit": np.roll(x, 19),
    }


def test_teacher_student_hyperparameters_are_frozen():
    assert DEFAULT_TEACHER_STUDENT_PARAMS == {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "max_depth": 5,
        "min_samples_leaf": 100,
        "l2_regularization": 5.0,
        "early_stopping": False,
        "random_state": 20260811,
    }


def test_held_teacher_values_cannot_change_same_fold_student_predictions():
    n = 120
    folds = np.repeat(np.arange(5), n // 5)
    categories = np.array([f"C{i % 4}" for i in range(n)], dtype=object)
    teacher = np.sin(np.linspace(0.0, 8.0, n)) * 3.0
    first = crossfit_teacher_student(_scores(n), teacher, categories, folds)

    changed_teacher = teacher.copy()
    held = folds == 2
    changed_teacher[held] += np.linspace(1000.0, 9000.0, held.sum())
    second = crossfit_teacher_student(_scores(n), changed_teacher, categories, folds)
    np.testing.assert_allclose(
        first["oof_score"][held], second["oof_score"][held], atol=0.0, rtol=0.0
    )


def test_teacher_student_scores_every_row_and_is_finite():
    n = 120
    folds = np.repeat(np.arange(5), n // 5)
    categories = np.array([f"C{i % 4}" for i in range(n)], dtype=object)
    teacher = np.cos(np.linspace(0.0, 6.0, n))
    result = crossfit_teacher_student(_scores(n), teacher, categories, folds)
    score = result["oof_score"]
    assert score.shape == (n,)
    assert np.isfinite(score).all()
    assert ((score >= 0.0) & (score <= 1.0)).all()
    assert set(result["fold_models"]) == {0, 1, 2, 3, 4}
