import numpy as np
import torch

from aios_track2.surrogates.base import EnsemblePrediction
from aios_track2.surrogates.graph import GraphTemporalSurrogate, normalize_adjacency
from aios_track2.surrogates.gru import GRUSurrogate
from aios_track2.surrogates.linear import LinearSurrogate
from aios_track2.surrogates.tcn import TCNSurrogate


def test_linear_surrogate_fits_multivariate_relationship() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 5, 3))
    y = np.stack([2 * x[..., 0] - x[..., 1], x[..., 2] + 0.5], axis=-1)
    model = LinearSurrogate().fit(x, y)
    pred = model.predict(x)
    assert pred.mean.shape == y.shape
    assert pred.std.shape == y.shape
    assert np.mean(np.abs(pred.mean - y)) < 1e-3


def test_temporal_models_keep_batch_time_well_axes() -> None:
    x = torch.randn(2, 7, 5, 4)
    gru = GRUSurrogate(4, 3, hidden=12)
    tcn = TCNSurrogate(4, 3, hidden=12)
    assert gru(x).shape == (2, 7, 5, 3)
    assert tcn(x).shape == (2, 7, 5, 3)


def test_gru_output_is_contiguous_for_mps_losses() -> None:
    x = torch.randn(2, 7, 5, 4)
    output = GRUSurrogate(4, 3, hidden=12)(x)
    assert output.is_contiguous()


def test_graph_temporal_model_couples_wells() -> None:
    adjacency = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    normalized = normalize_adjacency(adjacency)
    assert torch.allclose(normalized, normalized.T)
    model = GraphTemporalSurrogate(2, 4, adjacency=adjacency, hidden=10)
    x = torch.randn(3, 6, 3, 2)
    assert model(x).shape == (3, 6, 3, 4)


def test_ensemble_prediction_exposes_epistemic_std() -> None:
    a = np.array([[1.0, 2.0]])
    b = np.array([[3.0, 4.0]])
    prediction = EnsemblePrediction.from_members([a, b])
    np.testing.assert_allclose(prediction.mean, [[2.0, 3.0]])
    np.testing.assert_allclose(prediction.std, [[1.0, 1.0]])
