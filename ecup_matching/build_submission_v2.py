from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


METADATA = {
    "image": "odsai/ecup26-matching-baseline:1.0",
    "entry_point": "python -u run.py",
}
MAX_ARCHIVE_BYTES = 5 * 1024**3
RUNTIME_FILES = (
    "ecup_matching/ml/__init__.py",
    "ecup_matching/ml/textnorm.py",
    "ecup_matching/ml/features.py",
    "ecup_matching/ml/category_attrs.py",
    "ecup_matching/ml/features_v2.py",
    "ecup_matching/ml/model_io.py",
    "ecup_matching/submission/__init__.py",
    "ecup_matching/submission/predict_v2.py",
)


def build_submission_v2(model_path: Path, manifest_path: Path, output_path: Path) -> Path:
    model_path = Path(model_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise ValueError("v2 model file is missing or empty")
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise ValueError("v2 manifest file is missing or empty")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "v2-2024-transfer-structured":
        raise ValueError("unexpected v2 model manifest version")
    if not isinstance(manifest.get("attribute_importance"), dict):
        raise ValueError("v2 manifest is missing attribute_importance")

    repo_root = Path(__file__).resolve().parents[1]
    run_source = repo_root / "ecup_matching/submission/run_v2.py"
    if not run_source.is_file():
        raise FileNotFoundError(run_source)
    for rel in RUNTIME_FILES:
        if not (repo_root / rel).is_file():
            raise FileNotFoundError(repo_root / rel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("metadata.json", json.dumps(METADATA, ensure_ascii=False, indent=2) + "\n")
        zf.writestr("ecup_matching/__init__.py", '"""E-CUP submission runtime package."""\n')
        zf.write(run_source, "run.py")
        for rel in RUNTIME_FILES:
            zf.write(repo_root / rel, rel)
        zf.write(model_path, "model_v2.joblib")
        zf.write(manifest_path, "model_v2_manifest.json")

    if output_path.stat().st_size >= MAX_ARCHIVE_BYTES:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("v2 submission archive exceeds 5 GiB limit")

    with zipfile.ZipFile(output_path) as zf:
        names = zf.namelist()
        required = {
            "metadata.json",
            "run.py",
            "model_v2.joblib",
            "model_v2_manifest.json",
            "ecup_matching/submission/predict_v2.py",
            "ecup_matching/ml/features_v2.py",
            "ecup_matching/ml/category_attrs.py",
        }
        missing = required - set(names)
        if missing:
            raise RuntimeError(f"v2 submission archive is missing files: {sorted(missing)}")
        forbidden_suffixes = (".parquet", ".db", ".pem", ".b64", ".zip")
        forbidden = [
            name
            for name in names
            if name.endswith(forbidden_suffixes)
            or name.startswith("__MACOSX/")
            or "token" in name.lower()
            or "secret" in name.lower()
        ]
        if forbidden:
            raise RuntimeError(f"forbidden files in v2 submission: {forbidden}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = build_submission_v2(args.model, args.manifest, args.output)
    print(f"Built {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
