import numpy as np
import pytest

from ecup_matching.ml.metrics import OFFICIAL_CATEGORIES, macro_average_precision


def test_macro_average_precision_is_unweighted_mean_of_categories():
    y_true = np.array([1, 0, 1, 0, 0, 1])
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.6])
    categories = np.array(["a", "a", "b", "b", "b", "b"])

    macro, per_category = macro_average_precision(y_true, y_score, categories)

    assert set(per_category) == {"a", "b"}
    assert 0.0 <= macro <= 1.0
    assert macro == np.mean(list(per_category.values()))


def test_macro_average_precision_keeps_continuous_ranking():
    y_true = np.array([1, 0, 1, 0])
    categories = np.array(["a", "a", "a", "a"])
    good = np.array([0.9, 0.8, 0.7, 0.1])
    bad = np.array([0.8, 0.9, 0.1, 0.7])

    good_score, _ = macro_average_precision(y_true, good, categories)
    bad_score, _ = macro_average_precision(y_true, bad, categories)
    assert good_score > bad_score


def test_strict_metric_rejects_missing_official_category():
    present = OFFICIAL_CATEGORIES[:-1]
    categories = np.repeat(np.asarray(present, dtype=object), 2)
    y_true = np.tile(np.array([0, 1], dtype=np.int8), len(present))
    y_score = np.tile(np.array([0.1, 0.9], dtype=float), len(present))

    with pytest.raises(ValueError, match="category set"):
        macro_average_precision(
            y_true,
            y_score,
            categories,
            expected_categories=OFFICIAL_CATEGORIES,
            require_both_classes=True,
        )


def test_strict_metric_rejects_single_class_category():
    categories = np.repeat(np.asarray(OFFICIAL_CATEGORIES, dtype=object), 2)
    y_true = np.tile(np.array([0, 1], dtype=np.int8), len(OFFICIAL_CATEGORIES))
    y_true[:2] = 1
    y_score = np.linspace(0.01, 0.99, len(categories))

    with pytest.raises(ValueError, match="both target classes"):
        macro_average_precision(
            y_true,
            y_score,
            categories,
            expected_categories=OFFICIAL_CATEGORIES,
            require_both_classes=True,
        )
