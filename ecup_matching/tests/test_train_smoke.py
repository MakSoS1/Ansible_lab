import numpy as np
import pandas as pd

from ecup_matching.ml.model_io import load_model_bundle, save_model_bundle
from ecup_matching.ml.train_v1 import train_estimator


def test_v1_estimator_trains_roundtrips_and_predicts_probabilities(tmp_path):
    rng = np.random.default_rng(2026)
    n = 80
    frame = pd.DataFrame(
        {
            "category": np.where(np.arange(n) % 2, "phones", "care"),
            "same_category": np.ones(n),
            "name_exact": np.r_[np.ones(n // 2), np.zeros(n // 2)],
            "fuzz_ratio": np.r_[rng.uniform(0.85, 1.0, n // 2), rng.uniform(0.1, 0.7, n // 2)],
        }
    )
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)

    model = train_estimator(frame, y, sample_weight=np.ones(n), max_iter=40)
    probs = model.predict_proba(frame)[:, 1]
    assert np.isfinite(probs).all()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert probs[: n // 2].mean() > probs[n // 2 :].mean()

    model_path = tmp_path / "model.joblib"
    manifest_path = tmp_path / "manifest.json"
    save_model_bundle(model, model_path, manifest_path, {"version": "v1"})
    restored, manifest = load_model_bundle(model_path, manifest_path)
    assert manifest["version"] == "v1"
    np.testing.assert_allclose(restored.predict_proba(frame)[:, 1], probs)
