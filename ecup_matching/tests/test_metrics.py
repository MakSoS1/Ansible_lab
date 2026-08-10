import numpy as np

from ecup_matching.ml.metrics import macro_average_precision


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
