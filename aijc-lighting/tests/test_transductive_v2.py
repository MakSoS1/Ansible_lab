import numpy as np

from src.transductive import balanced_assignment, evaluate_count_constraint


def test_balanced_assignment_obeys_exact_counts():
    probs = np.array([
        [.9, .05, .05], [.8, .1, .1], [.1, .8, .1],
        [.1, .7, .2], [.1, .2, .7], [.1, .1, .8],
    ])
    pred = balanced_assignment(probs, (2, 2, 2))
    assert np.bincount(pred, minlength=3).tolist() == [2, 2, 2]


def test_evaluate_count_constraint_reports_both_paths():
    y = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array([
        [.6, .3, .1], [.55, .4, .05], [.45, .5, .05],
        [.42, .4, .18], [.1, .1, .8], [.1, .2, .7],
    ])
    report = evaluate_count_constraint(probs, y)
    assert report["constrained_counts"] == [2, 2, 2]
    assert report["constrained_accuracy"] >= report["argmax_accuracy"]
