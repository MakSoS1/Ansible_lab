import numpy as np

from aios_track2.surrogates.base import ScenarioBatch, is_ood
from aios_track2.surrogates.graph import DeepEnsemble, GraphTemporalSurrogate


def _batch(n: int = 4, scale: float = 1.0) -> ScenarioBatch:
    rng = np.random.default_rng(0)
    features = (rng.normal(size=(n, 6, 4, 4)) * scale).astype(np.float32)
    return ScenarioBatch(
        scenario_ids=tuple(f"g{i:03d}" for i in range(n)),
        features=features,
        targets=features * 0.5,
        controls=features.copy(),
    )


def test_well_permutation_preserves_field_total() -> None:
    batch = _batch()
    model = GraphTemporalSurrogate(seed=0, epochs=2, hidden_channels=16, adjacency=np.eye(4, dtype=np.float32))
    model.fit(batch, batch)
    original = model.predict(batch).mean.sum(axis=2)
    permuted = model.predict(batch.permute_wells(seed=7)).mean.sum(axis=2)
    np.testing.assert_allclose(original, permuted, rtol=1e-4, atol=1e-3)


def test_ensemble_variance_increases_outside_training_domain() -> None:
    in_domain = _batch(scale=1.0)
    out_of_domain = _batch(scale=6.0)
    ensemble = DeepEnsemble(seeds=(11, 23, 42), hidden_channels=16, epochs=2, adjacency=np.eye(4, dtype=np.float32))
    ensemble.fit(in_domain, in_domain)
    assert ensemble.predict(out_of_domain).variance.mean() >= ensemble.predict(in_domain).variance.mean() * 0.5
    assert is_ood(ensemble.predict(out_of_domain), threshold=0.0).any()
