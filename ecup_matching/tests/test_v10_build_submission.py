import json
from pathlib import Path
import zipfile

from ecup_matching.build_submission_v10 import build_submission_v10
from ecup_matching.submission.v10_contract import assert_student_only_archive


def _fake_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "bert"}), encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"tiny-weights")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    return model


def test_v10_builder_emits_one_student_and_runtime_closure(tmp_path):
    model = _fake_model(tmp_path)
    output = tmp_path / "v10.zip"
    report = build_submission_v10(
        model_dir=model,
        output_zip=output,
        strict_oof=0.55,
        base_model_revision="e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae",
        max_length=128,
        max_chars=650,
        inference_batch_size=128,
    )
    assert output.is_file()
    contract = assert_student_only_archive(output, max_bytes=150_000_000)
    assert contract["model_weight_files"] == 1
    assert report["model_metadata"]["version"] == "v10-tiny-student"
    assert report["model_metadata"]["strict_oof_macro_average_precision"] == 0.55

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "run.py" in names
    assert "model_v10_student/model.safetensors" in names
    assert "model_v10_metadata.json" in names
    assert "ecup_matching/submission/predict_v10.py" in names
    assert "ecup_matching/ml/v7_runtime.py" in names
    assert not any("model_v9" in name.lower() for name in names)
