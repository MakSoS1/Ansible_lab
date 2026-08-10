import json
import zipfile
from pathlib import Path

import numpy as np

from ecup_matching.build_submission_v3 import build_submission_v3
from ecup_matching.submission.predict_v3 import apply_category_blend, categories_requiring_neural


def test_categories_requiring_neural_uses_only_positive_alphas():
    manifest = {"category_alphas": {"A": 0.0, "B": 0.45, "C": 1.0}}
    assert categories_requiring_neural(manifest) == {"B", "C"}


def test_categories_requiring_neural_marks_global_blend():
    manifest = {"category_alphas": {"__global__": 0.30}}
    assert categories_requiring_neural(manifest) == {"*"}


def test_apply_category_blend_preserves_structured_rows_with_zero_alpha():
    categories = np.array(["A", "B", "C", "B"], dtype=object)
    structured = np.array([0.1, 0.2, 0.3, 0.4])
    neural = np.array([0.9, 0.8, 0.7, 0.6])
    manifest = {"category_alphas": {"A": 0.0, "B": 0.5, "C": 1.0}}
    out = apply_category_blend(categories, structured, neural, manifest)
    np.testing.assert_allclose(out, [0.1, 0.5, 0.7, 0.5])
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_apply_category_blend_supports_global_alpha():
    categories = np.array(["A", "B"], dtype=object)
    structured = np.array([0.2, 0.4])
    neural = np.array([0.8, 0.6])
    manifest = {"category_alphas": {"__global__": 0.25}}
    out = apply_category_blend(categories, structured, neural, manifest)
    np.testing.assert_allclose(out, [0.35, 0.45])


def test_build_submission_v3_contains_only_offline_runtime_and_models(tmp_path: Path):
    structured_model = tmp_path / "model_v2.joblib"
    structured_manifest = tmp_path / "model_v2_manifest.json"
    neural_model = tmp_path / "model"
    neural_manifest = tmp_path / "v3_manifest.json"
    structured_model.write_bytes(b"joblib-placeholder")
    structured_manifest.write_text(
        json.dumps({
            "version": "v2-2024-transfer-structured",
            "feature_names": [],
            "attribute_importance": {},
        }),
        encoding="utf-8",
    )
    neural_model.mkdir()
    (neural_model / "config.json").write_text("{}", encoding="utf-8")
    (neural_model / "model.safetensors").write_bytes(b"safe")
    (neural_model / "tokenizer.json").write_text("{}", encoding="utf-8")
    neural_manifest.write_text(
        json.dumps({
            "version": "v3-compact-reranker",
            "base_model": "cointegrated/rubert-tiny2",
            "max_length": 160,
            "category_alphas": {"Электроника": 0.5},
        }),
        encoding="utf-8",
    )
    output = tmp_path / "v3.zip"

    built = build_submission_v3(
        structured_model_path=structured_model,
        structured_manifest_path=structured_manifest,
        neural_model_dir=neural_model,
        neural_manifest_path=neural_manifest,
        output_path=output,
    )

    assert built == output
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        assert "metadata.json" in names
        assert "run.py" in names
        assert "model_v2.joblib" in names
        assert "model_v2_manifest.json" in names
        assert "model_v3_manifest.json" in names
        assert "model_v3/model.safetensors" in names
        assert "model_v3/tokenizer.json" in names
        assert "ecup_matching/submission/predict_v3.py" in names
        assert not any(name.endswith((".parquet", ".db", ".pem", ".b64")) for name in names)
        suspicious = ("hf_token", "api_token", "access_token", "secret", "password", "credential")
        assert not any(any(term in name.lower() for term in suspicious) for name in names)
