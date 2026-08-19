import json

import numpy as np
import pandas as pd

from ecup_matching.ml.features import build_pair_features
from ecup_matching.ml.model_io import save_model_bundle
from ecup_matching.ml.train_v1 import train_estimator
from ecup_matching.submission.predict import predict_to_csv


def test_submission_preserves_order_and_writes_exact_schema(tmp_path):
    items = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["Phone A15", "Phone A15", "Phone A16", "Soap 500 ml", "Soap 1 l", "Soap 500 ml"],
            "attributes": [json.dumps({"brand": "A"})] * 6,
            "category": ["phones", "phones", "phones", "care", "care", "care"],
        }
    )
    train_pairs = pd.DataFrame(
        {
            "id1": [1, 1, 4, 4, 2, 5] * 8,
            "id2": [2, 3, 6, 5, 3, 6] * 8,
            "target": [1, 0, 1, 0, 0, 0] * 8,
        }
    )
    x = build_pair_features(items, train_pairs)
    model = train_estimator(x, train_pairs["target"].to_numpy(), max_iter=30)
    model_path = tmp_path / "model.joblib"
    manifest_path = tmp_path / "manifest.json"
    save_model_bundle(model, model_path, manifest_path, {"version": "test"})

    test_pairs = pd.DataFrame({"id1": [4, 1, 1], "id2": [6, 3, 2]})
    items_path = tmp_path / "items.parquet"
    pairs_path = tmp_path / "pairs.parquet"
    out_path = tmp_path / "prediction.csv"
    items.to_parquet(items_path, index=False)
    test_pairs.to_parquet(pairs_path, index=False)

    result = predict_to_csv(items_path, pairs_path, model_path, manifest_path, out_path, chunk_size=2)

    assert list(result.columns) == ["id1", "id2", "predict"]
    assert result[["id1", "id2"]].values.tolist() == test_pairs[["id1", "id2"]].values.tolist()
    assert np.isfinite(result["predict"]).all()
    assert result["predict"].between(0, 1).all()
    disk = pd.read_csv(out_path)
    assert list(disk.columns) == ["id1", "id2", "predict"]
    assert len(disk) == 3
