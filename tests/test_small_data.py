import numpy as np
import pytest

from aios_track2.small_data import AdditiveGroupKernelRidge, QuadraticRidge, StationaryKernelRidge, project_temporal_policy


def test_quadratic_ridge_recovers_quadratic_function() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(0.8, 1.2, size=(50, 4))
    z = x - 1.0
    y = 3.0 + z[:, 0] - 2 * z[:, 1] + 4 * z[:, 2] * z[:, 3]
    model = QuadraticRidge(ridge=1e-10).fit(x, y)
    assert np.max(np.abs(model.predict(x) - y)) < 1e-6


def test_stationary_kernel_roundtrip_shape() -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(0.8, 1.2, size=(16, 6))
    y = rng.normal(size=(16, 5, 2))
    model = StationaryKernelRidge(kind="matern52", length_scale=2.0).fit(x, y)
    assert model.predict(x[:3]).shape == (3, 5, 2)


def test_additive_group_kernel_learns_group_additive_response() -> None:
    rng = np.random.default_rng(9)
    x = rng.uniform(0.8, 1.2, size=(40, 18))
    z = x.reshape(40, 6, 3) - 1.0
    y = np.sum(np.sin(4 * z) + 0.2 * z**2, axis=(1, 2))
    model = AdditiveGroupKernelRidge(group_size=3, length_scale=1.5, ridge=1e-6, global_weight=0.0).fit(x, y)
    pred = model.predict(x)
    assert np.corrcoef(y, pred)[0, 1] > 0.999


def test_temporal_projection_enforces_bounds_and_adjacent_delta() -> None:
    values = np.asarray([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    projected = project_temporal_policy(values, groups=2, nodes=3, lower=0.8, upper=1.2, max_delta=0.12)
    matrix = projected.reshape(1, 2, 3)
    assert np.all((matrix >= 0.8) & (matrix <= 1.2))
    assert np.max(np.abs(np.diff(matrix, axis=2))) <= 0.1200001


def test_invalid_group_width_fails_closed() -> None:
    with pytest.raises(ValueError):
        AdditiveGroupKernelRidge(group_size=4).fit(np.zeros((2, 6)), np.zeros(2))
