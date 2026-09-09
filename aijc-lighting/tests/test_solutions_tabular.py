import numpy as np
import pandas as pd

from src.solutions import run_solution_1, run_solution_2, run_solution_3


def _linear_dataset(seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), 30)
    x = y[:, None] * 5.0 + rng.normal(0, 0.35, size=(90, 4))
    xt = np.vstack([
        np.full((5, 4), 0.0),
        np.full((5, 4), 5.0),
        np.full((5, 4), 10.0),
    ]) + rng.normal(0, 0.1, size=(15, 4))
    return pd.DataFrame(x), pd.DataFrame(xt), y


def _assert_result(result, n=90, m=15):
    assert result.oof_probs.shape == (n, 3)
    assert result.test_probs.shape == (m, 3)
    assert np.allclose(result.oof_probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(result.test_probs.sum(axis=1), 1.0, atol=1e-5)
    assert 0.0 <= result.oof_accuracy <= 1.0


def test_solution_1_solves_clear_luminance_bands():
    X, Xt, y = _linear_dataset()
    result = run_solution_1(X, Xt, y, n_splits=3, seed=42)
    _assert_result(result)
    assert result.oof_accuracy > 0.95


def test_solution_2_probability_contract():
    X, Xt, y = _linear_dataset(1)
    result = run_solution_2(X, Xt, y, n_splits=3, seed=42, fast=True)
    _assert_result(result)
    assert result.oof_accuracy > 0.85


def test_solution_3_probability_contract():
    X, Xt, y = _linear_dataset(2)
    result = run_solution_3(X, Xt, y, n_splits=3, seed=42, fast=True)
    _assert_result(result)
    assert result.oof_accuracy > 0.85
