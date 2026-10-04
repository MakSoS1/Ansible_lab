import numpy as np

from aios_track2.surrogates.base import ScenarioBatch, evaluate_surrogate
from aios_track2.surrogates.linear import LinearSurrogate
from aios_track2.surrogates.tcn import TCNSurrogate


def _batch(n: int = 6, seed: int = 0) -> ScenarioBatch:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 8, 3, 4)).astype(np.float32)
    targets = features * 0.4 + 0.1
    return ScenarioBatch(
        scenario_ids=tuple(f"s{i:03d}" for i in range(n)),
        features=features,
        targets=targets,
        controls=features.copy(),
    )


def test_prediction_contains_finite_mean_and_variance() -> None:
    batch = _batch()
    model = LinearSurrogate(seed=42).fit(batch, batch)
    prediction = model.predict(batch)
    assert prediction.mean.shape == batch.targets.shape
    assert prediction.variance.shape == batch.targets.shape
    assert np.isfinite(prediction.mean).all()
    assert (prediction.variance >= 0).all()


def test_evaluator_rejects_train_ids() -> None:
    batch = _batch()
    model = LinearSurrogate(seed=1)
    model.fit(batch, batch)
    try:
        evaluate_surrogate(model, batch, train_ids=batch.scenario_ids)
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("leakage was not detected")


def test_tcn_predicts() -> None:
    batch = _batch(n=4)
    model = TCNSurrogate(seed=0, epochs=2, hidden_channels=16)
    model.fit(batch, batch)
    prediction = model.predict(batch)
    assert prediction.mean.shape == batch.targets.shape
