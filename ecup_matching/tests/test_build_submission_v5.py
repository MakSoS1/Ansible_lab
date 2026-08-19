import json
import zipfile
from pathlib import Path

from ecup_matching.build_submission_v5 import build_submission_v5


def _fake_model_dir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"model")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


def test_build_submission_v5_contains_offline_runtime_and_models(tmp_path):
    structured = tmp_path / "structured.joblib"
    structured.write_bytes(b"structured")
    contrastive = _fake_model_dir(tmp_path, "contrastive")
    teacher = _fake_model_dir(tmp_path, "teacher")
    legacy = tmp_path / "legacy_ecup"
    (legacy / "ml").mkdir(parents=True)
    (legacy / "__init__.py").write_text("", encoding="utf-8")
    (legacy / "ml" / "__init__.py").write_text("", encoding="utf-8")
    (legacy / "ml" / "textnorm.py").write_text("VALUE=1\n", encoding="utf-8")
    output = tmp_path / "submit.zip"

    build_submission_v5(
        structured_model_path=structured,
        contrastive_model_dir=contrastive,
        teacher_model_dir=teacher,
        legacy_runtime_dir=legacy,
        output_path=output,
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata == {
        "image": "odsai/ecup26-matching-baseline:1.0",
        "entry_point": "python -u run.py",
    }
    assert "run.py" in names
    assert "model_v5_structured.joblib" in names
    assert "model_v5_contrastive/config.json" in names
    assert "model_v5_teacher/config.json" in names
    assert "legacy_ecup/ml/textnorm.py" in names
    assert "ecup_matching/submission/predict_v5.py" in names
    assert not any(name.endswith(".parquet") for name in names)
