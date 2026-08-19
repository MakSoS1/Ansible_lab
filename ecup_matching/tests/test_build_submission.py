import json
import zipfile

from ecup_matching.build_submission import build_submission


def test_submission_builder_has_official_root_contract_and_no_raw_data(tmp_path):
    model = tmp_path / "model.joblib"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "submit.zip"
    model.write_bytes(b"fake-model")
    manifest.write_text(json.dumps({"version": "test", "feature_names": []}), encoding="utf-8")

    build_submission(model, manifest, output)

    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        metadata = json.loads(zf.read("metadata.json"))
    assert metadata == {
        "image": "odsai/ecup26-matching-baseline:1.0",
        "entry_point": "python -u run.py",
    }
    assert "run.py" in names
    assert "model_v1.joblib" in names
    assert "model_v1_manifest.json" in names
    assert "ecup_matching/submission/predict.py" in names
    assert "ecup_matching/ml/features.py" in names
    assert not any(name.endswith(".parquet") for name in names)
    assert not any("HF_TOKEN" in name for name in names)
