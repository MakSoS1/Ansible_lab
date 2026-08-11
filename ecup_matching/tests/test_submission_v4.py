import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ecup_matching.build_submission_v4 import build_submission_v4
from ecup_matching.submission.predict_v3 import apply_category_blend, categories_requiring_neural


def _structured_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "v2-2024-transfer-structured",
                "feature_names": [],
                "attribute_importance": {},
            }
        ),
        encoding="utf-8",
    )


def _neural_manifest(path: Path, *, selected_blend: str = "global") -> None:
    path.write_text(
        json.dumps(
            {
                "version": "v4-strong-reranker",
                "base_model": "ai-forever/ruBert-base",
                "base_model_revision": "43be4261797042e172adf7476c558734f3cbb2a0",
                "selected_model_stage": "v4b",
                "selected_blend": selected_blend,
                "validation_macro_ap": 0.54,
                "max_length": 256,
                "max_attrs": 10,
                "max_chars": 700,
                "category_alphas": {"__global__": 0.55},
            }
        ),
        encoding="utf-8",
    )


def test_v4_global_manifest_routes_all_pairs_to_neural() -> None:
    manifest = {"category_alphas": {"__global__": 0.55}}
    assert categories_requiring_neural(manifest) == {"*"}
    out = apply_category_blend(
        np.array(["A", "B"], dtype=object),
        np.array([0.2, 0.4]),
        np.array([0.8, 0.6]),
        manifest,
    )
    np.testing.assert_allclose(out, [0.53, 0.51])


def test_build_submission_v4_contains_exact_offline_runtime_and_models(tmp_path: Path) -> None:
    structured_model = tmp_path / "model.joblib"
    structured_manifest = tmp_path / "structured.json"
    neural_model = tmp_path / "model"
    neural_manifest = tmp_path / "manifest.json"
    structured_model.write_bytes(b"joblib-placeholder")
    _structured_manifest(structured_manifest)
    neural_model.mkdir()
    (neural_model / "config.json").write_text("{}", encoding="utf-8")
    (neural_model / "model.safetensors").write_bytes(b"safe")
    (neural_model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (neural_model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    _neural_manifest(neural_manifest)
    output = tmp_path / "v4.zip"

    built = build_submission_v4(
        structured_model_path=structured_model,
        structured_manifest_path=structured_manifest,
        neural_model_dir=neural_model,
        neural_manifest_path=neural_manifest,
        output_path=output,
    )

    assert built == output
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "metadata.json" in names
        assert "run.py" in names
        assert "model_v2.joblib" in names
        assert "model_v2_manifest.json" in names
        assert "model_v4_manifest.json" in names
        assert "model_v4/model.safetensors" in names
        assert "model_v4/tokenizer.json" in names
        assert "ecup_matching/submission/predict_v4.py" in names
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata == {
            "image": "odsai/ecup26-matching-baseline:1.0",
            "entry_point": "python -u run.py",
        }
        forbidden_suffixes = (".parquet", ".db", ".pem", ".b64")
        assert not any(name.endswith(forbidden_suffixes) for name in names)
        suspicious = (
            "hf_token",
            "api_token",
            "access_token",
            "secret",
            "password",
            "credential",
        )
        assert not any(any(term in name.lower() for term in suspicious) for name in names)


def test_build_submission_v4_rejects_manifest_without_positive_neural_route(tmp_path: Path) -> None:
    structured_model = tmp_path / "model.joblib"
    structured_manifest = tmp_path / "structured.json"
    neural_model = tmp_path / "model"
    neural_manifest = tmp_path / "manifest.json"
    structured_model.write_bytes(b"joblib-placeholder")
    _structured_manifest(structured_manifest)
    neural_model.mkdir()
    (neural_model / "config.json").write_text("{}", encoding="utf-8")
    (neural_model / "model.safetensors").write_bytes(b"safe")
    neural_manifest.write_text(
        json.dumps(
            {
                "version": "v4-strong-reranker",
                "base_model": "ai-forever/ruBert-base",
                "base_model_revision": "43be4261797042e172adf7476c558734f3cbb2a0",
                "category_alphas": {"__global__": 0.0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="zero"):
        build_submission_v4(
            structured_model_path=structured_model,
            structured_manifest_path=structured_manifest,
            neural_model_dir=neural_model,
            neural_manifest_path=neural_manifest,
            output_path=tmp_path / "v4.zip",
        )
