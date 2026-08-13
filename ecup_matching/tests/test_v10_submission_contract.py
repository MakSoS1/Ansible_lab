from pathlib import Path
import zipfile

import pytest

from ecup_matching.submission.v10_contract import assert_student_only_archive


def _write_zip(path: Path, members: list[str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.writestr(member, "x")


def test_student_only_archive_accepts_minimal_tiny_student(tmp_path):
    archive = tmp_path / "v10.zip"
    _write_zip(
        archive,
        [
            "run.py",
            "predict.py",
            "model/config.json",
            "model/model.safetensors",
            "model/tokenizer.json",
            "model/tokenizer_config.json",
            "model/special_tokens_map.json",
            "graph_config.json",
        ],
    )
    report = assert_student_only_archive(archive, max_bytes=150_000_000)
    assert report["forbidden_members"] == []
    assert report["model_weight_files"] == 1


def test_student_only_archive_rejects_v9_heavy_components(tmp_path):
    archive = tmp_path / "bad.zip"
    _write_zip(
        archive,
        [
            "run.py",
            "model/model.safetensors",
            "models/teacher/model.safetensors",
            "models/contrastive/model.safetensors",
            "structured_model.joblib",
        ],
    )
    with pytest.raises(ValueError, match="forbidden"):
        assert_student_only_archive(archive, max_bytes=150_000_000)


def test_student_only_archive_rejects_multiple_weight_files_and_oversize(tmp_path):
    archive = tmp_path / "two-models.zip"
    _write_zip(
        archive,
        ["run.py", "model/model.safetensors", "model/extra.safetensors"],
    )
    with pytest.raises(ValueError, match="exactly one"):
        assert_student_only_archive(archive, max_bytes=150_000_000)

    tiny_limit = archive.stat().st_size - 1
    with pytest.raises(ValueError, match="size"):
        assert_student_only_archive(archive, max_bytes=tiny_limit)
