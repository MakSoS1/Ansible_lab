from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np


def _write_safetensors(path: Path) -> None:
    from safetensors.numpy import save_file

    save_file(
        {
            "weight": np.asarray([[1.25, -2.5], [3.0, 0.125]], dtype=np.float32),
            "already_half": np.asarray([0.5, -0.25], dtype=np.float16),
            "ids": np.asarray([1, 7, 42], dtype=np.int64),
        },
        str(path),
        metadata={"format": "pt", "fixture": "v9-compact"},
    )


def test_convert_safetensors_fp16_only_downcasts_wide_float_tensors(tmp_path: Path) -> None:
    from safetensors import safe_open
    from safetensors.numpy import load_file

    from ecup_matching.submission.build_submission_v9_compact import convert_safetensors_fp16

    src = tmp_path / "source.safetensors"
    dst = tmp_path / "compact.safetensors"
    _write_safetensors(src)

    report = convert_safetensors_fp16(src, dst)
    tensors = load_file(str(dst))

    assert tensors["weight"].dtype == np.float16
    assert tensors["already_half"].dtype == np.float16
    assert tensors["ids"].dtype == np.int64
    np.testing.assert_array_equal(tensors["ids"], np.asarray([1, 7, 42], dtype=np.int64))
    np.testing.assert_allclose(
        tensors["weight"].astype(np.float32),
        np.asarray([[1.25, -2.5], [3.0, 0.125]], dtype=np.float32),
        rtol=0,
        atol=0,
    )
    with safe_open(str(dst), framework="np") as handle:
        assert handle.metadata() == {"format": "pt", "fixture": "v9-compact"}
    assert report["converted_tensors"] == ["weight"]
    assert report["preserved_tensors"] == ["already_half", "ids"]
    assert report["source_bytes"] == src.stat().st_size
    assert report["output_bytes"] == dst.stat().st_size


def test_compact_v9_submission_keeps_runtime_and_models_but_removes_bytecode(tmp_path: Path) -> None:
    from safetensors.numpy import load_file

    from ecup_matching.submission.build_submission_v9_compact import compact_v9_submission

    tree = tmp_path / "tree"
    (tree / "model_v5_teacher").mkdir(parents=True)
    (tree / "model_v5_contrastive").mkdir(parents=True)
    (tree / "ecup_matching/submission/__pycache__").mkdir(parents=True)
    (tree / "ecup_matching/submission").mkdir(parents=True, exist_ok=True)
    (tree / "ecup_matching/ml").mkdir(parents=True)

    (tree / "run.py").write_text("print('runtime')\n", encoding="utf-8")
    (tree / "metadata.json").write_text("{}\n", encoding="utf-8")
    (tree / "model_v5_structured.joblib").write_bytes(b"structured")
    (tree / "model_v6_category_shrunk.json").write_text("{}\n", encoding="utf-8")
    (tree / "model_v6_hgb_meta.joblib").write_bytes(b"hgb")
    (tree / "model_v6_gate_metadata.json").write_text(
        json.dumps({"version": "v9-gate40-production-refit", "coverage": 0.4}),
        encoding="utf-8",
    )
    (tree / "v9_metadata.json").write_text(
        json.dumps({"version": "v9-gate40-fp16-graph"}), encoding="utf-8"
    )
    (tree / "ecup_matching/submission/predict_v6.py").write_text("X=1\n", encoding="utf-8")
    (tree / "ecup_matching/ml/v8_submission_graph.py").write_text("X=1\n", encoding="utf-8")
    (tree / "ecup_matching/submission/__pycache__/predict_v6.cpython-311.pyc").write_bytes(b"dead")
    (tree / "model_v5_teacher/config.json").write_text("{}\n", encoding="utf-8")
    (tree / "model_v5_contrastive/config.json").write_text("{}\n", encoding="utf-8")
    _write_safetensors(tree / "model_v5_teacher/model.safetensors")
    _write_safetensors(tree / "model_v5_contrastive/model.safetensors")

    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(tree).as_posix())

    output = tmp_path / "compact.zip"
    report = compact_v9_submission(source_zip=source, output_zip=output, enforce_source_keeper=False)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "run.py" in names
        assert "model_v5_structured.joblib" in names
        assert "model_v5_teacher/model.safetensors" in names
        assert "model_v5_contrastive/model.safetensors" in names
        assert "ecup_matching/submission/predict_v6.py" in names
        assert "ecup_matching/ml/v8_submission_graph.py" in names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
        assert "v9_compact_metadata.json" in names
        compact_meta = json.loads(archive.read("v9_compact_metadata.json"))
        assert compact_meta["version"] == "v9-compact-fp16-storage"
        assert compact_meta["source_archive_sha256"] == report["source_archive_sha256"]

    extract = tmp_path / "extract"
    with zipfile.ZipFile(output) as archive:
        archive.extract("model_v5_teacher/model.safetensors", extract)
        archive.extract("model_v5_contrastive/model.safetensors", extract)
    assert load_file(str(extract / "model_v5_teacher/model.safetensors"))["weight"].dtype == np.float16
    assert load_file(str(extract / "model_v5_contrastive/model.safetensors"))["weight"].dtype == np.float16
    assert report["converted_model_files"] == [
        "model_v5_contrastive/model.safetensors",
        "model_v5_teacher/model.safetensors",
    ]
    assert report["removed_members"] == [
        "ecup_matching/submission/__pycache__/predict_v6.cpython-311.pyc"
    ]
