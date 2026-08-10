import json
import zipfile

import numpy as np
import pandas as pd

from ecup_matching.build_submission_v2 import build_submission_v2
from ecup_matching.ml.features_v2 import build_pair_features_v2
from ecup_matching.ml.model_io import save_model_bundle
from ecup_matching.ml.train_v2_structured import fit_estimator
from ecup_matching.submission.predict_v2 import predict_to_csv_v2


def _tiny_items():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "name": ["Phone A 128", "Phone A 128", "Phone A 256", "Phone B 64", "Phone B 64", "Phone B 128", "Shoe X 42", "Shoe X 43"],
        "attributes": [
            '{"brand":"a","memory":"128"}', '{"brand":"a","memory":"128"}', '{"brand":"a","memory":"256"}',
            '{"brand":"b","memory":"64"}', '{"brand":"b","memory":"64"}', '{"brand":"b","memory":"128"}',
            '{"brand":"x","size":"42"}', '{"brand":"x","size":"43"}',
        ],
        "category": ["electronics"] * 6 + ["shoes"] * 2,
    })


def _bundle(tmp_path):
    items = _tiny_items()
    train = pd.DataFrame({"id1": [1,1,4,4,7,7] * 6, "id2": [2,3,5,6,7,8] * 6})
    y = np.array([1,0,1,0,1,0] * 6, dtype=np.int8)
    importance = {"electronics": {"memory": 3.0, "brand": 1.0}, "shoes": {"size": 3.0, "brand": 1.0}}
    x = build_pair_features_v2(items, train, attribute_importance=importance)
    model = fit_estimator(x, y, np.ones(len(y)), max_iter=25)
    model_path = tmp_path / "model.joblib"
    manifest_path = tmp_path / "manifest.json"
    save_model_bundle(model, model_path, manifest_path, {
        "version": "v2-2024-transfer-structured",
        "selected_candidate": "v2b-weak-curriculum",
        "validation_macro_ap": 0.5010008994958702,
        "feature_names": list(x.columns),
        "attribute_importance": importance,
    })
    return items, model_path, manifest_path


def test_predict_v2_preserves_pair_order_schema_and_range(tmp_path):
    items, model_path, manifest_path = _bundle(tmp_path)
    matches = pd.DataFrame({"id1": [1,4,7], "id2": [3,5,8]})
    items_path = tmp_path / "items.parquet"
    matches_path = tmp_path / "matches.parquet"
    output_path = tmp_path / "prediction.csv"
    items.to_parquet(items_path, index=False)
    matches.to_parquet(matches_path, index=False)
    result = predict_to_csv_v2(items_path, matches_path, model_path, manifest_path, output_path, chunk_size=2)
    assert list(result.columns) == ["id1", "id2", "predict"]
    assert result[["id1", "id2"]].equals(matches)
    assert np.isfinite(result["predict"]).all()
    assert result["predict"].between(0, 1).all()
    assert output_path.is_file()


def test_build_v2_archive_is_offline_self_contained_and_clean(tmp_path):
    _, model_path, manifest_path = _bundle(tmp_path)
    output = tmp_path / "ecup-v2-submission.zip"
    build_submission_v2(model_path, manifest_path, output)
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        metadata = json.loads(zf.read("metadata.json"))
        assert metadata["image"] == "odsai/ecup26-matching-baseline:1.0"
        assert metadata["entry_point"] == "python -u run.py"
        required = {
            "run.py", "model_v2.joblib", "model_v2_manifest.json",
            "ecup_matching/submission/predict_v2.py",
            "ecup_matching/ml/features_v2.py", "ecup_matching/ml/category_attrs.py",
            "ecup_matching/ml/features.py", "ecup_matching/ml/textnorm.py",
        }
        assert required.issubset(names)
        assert not any(name.endswith((".parquet", ".db", ".pem", ".b64")) for name in names)
        assert not any("token" in name.lower() or "secret" in name.lower() for name in names)
