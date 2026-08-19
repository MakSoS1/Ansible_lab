from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


MAX_ZIP_BYTES = 5 * 1024**3
_METADATA = {
    "image": "odsai/ecup26-matching-baseline:1.0",
    "entry_point": "python -u run.py",
}
_PACKAGE_MARKERS = (
    "ecup_matching/__init__.py",
    "ecup_matching/ml/__init__.py",
    "ecup_matching/submission/__init__.py",
)
_RUNTIME_FILES = (
    "ecup_matching/ml/data_subset.py",
    "ecup_matching/ml/textnorm.py",
    "ecup_matching/ml/features.py",
    "ecup_matching/ml/category_attrs.py",
    "ecup_matching/ml/features_v2.py",
    "ecup_matching/ml/v5_category_specialists.py",
    "ecup_matching/ml/v5_explicit_attributes.py",
    "ecup_matching/ml/v5_fixed_blend.py",
    "ecup_matching/ml/v5_production.py",
    "ecup_matching/submission/predict_v5.py",
)
_FORBIDDEN_SUFFIXES = {".parquet", ".csv", ".pcap", ".env"}
_CREDENTIAL_BASENAMES = {
    "token", "token.txt", "hf_token", "hf_token.txt", "secret", "secrets", "credentials.json",
}


def _copy_tree_files(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in source.rglob("*"):
        if path.is_file():
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _validate_model_dir(path: Path, *, name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)
    if not (path / "config.json").is_file():
        raise ValueError(f"{name} model directory is missing config.json")
    if not any((path / candidate).is_file() for candidate in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError(f"{name} model directory is missing model weights")


def build_submission_v5(
    *,
    structured_model_path: Path,
    contrastive_model_dir: Path,
    teacher_model_dir: Path,
    legacy_runtime_dir: Path,
    output_path: Path,
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    structured_model_path = Path(structured_model_path)
    contrastive_model_dir = Path(contrastive_model_dir)
    teacher_model_dir = Path(teacher_model_dir)
    legacy_runtime_dir = Path(legacy_runtime_dir)
    output_path = Path(output_path)

    if not structured_model_path.is_file():
        raise FileNotFoundError(structured_model_path)
    _validate_model_dir(contrastive_model_dir, name="contrastive")
    _validate_model_dir(teacher_model_dir, name="teacher")
    if legacy_runtime_dir.name != "legacy_ecup" or not legacy_runtime_dir.is_dir():
        raise ValueError("legacy_runtime_dir must point to a legacy_ecup package")

    with tempfile.TemporaryDirectory(prefix="ecup-v5-submit-") as temp_dir:
        root = Path(temp_dir) / "submission"
        root.mkdir(parents=True)
        (root / "metadata.json").write_text(
            json.dumps(_METADATA, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        shutil.copy2(repo_root / "ecup_matching" / "submission" / "run_v5.py", root / "run.py")
        shutil.copy2(structured_model_path, root / "model_v5_structured.joblib")
        _copy_tree_files(contrastive_model_dir, root / "model_v5_contrastive")
        _copy_tree_files(teacher_model_dir, root / "model_v5_teacher")
        _copy_tree_files(legacy_runtime_dir, root / "legacy_ecup")

        for relative in _PACKAGE_MARKERS:
            source = repo_root / relative
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                shutil.copy2(source, target)
            else:
                target.write_text("", encoding="utf-8")

        for relative in _RUNTIME_FILES:
            source = repo_root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        forbidden = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and (
                path.suffix.lower() in _FORBIDDEN_SUFFIXES
                or path.name.lower() in _CREDENTIAL_BASENAMES
            )
        ]
        if forbidden:
            raise RuntimeError(f"forbidden files in submission tree: {forbidden[:5]}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())

    if output_path.stat().st_size > MAX_ZIP_BYTES:
        raise RuntimeError(
            f"submission ZIP exceeds 5 GiB: {output_path.stat().st_size} bytes"
        )
    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
        required = {
            "metadata.json",
            "run.py",
            "model_v5_structured.joblib",
            "model_v5_contrastive/config.json",
            "model_v5_teacher/config.json",
            "ecup_matching/submission/predict_v5.py",
            "legacy_ecup/ml/textnorm.py",
        }
        missing = required - names
        if missing:
            raise RuntimeError(f"submission ZIP missing files: {sorted(missing)}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-model", type=Path, required=True)
    parser.add_argument("--contrastive-model-dir", type=Path, required=True)
    parser.add_argument("--teacher-model-dir", type=Path, required=True)
    parser.add_argument("--legacy-runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = build_submission_v5(
        structured_model_path=args.structured_model,
        contrastive_model_dir=args.contrastive_model_dir,
        teacher_model_dir=args.teacher_model_dir,
        legacy_runtime_dir=args.legacy_runtime_dir,
        output_path=args.output,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
