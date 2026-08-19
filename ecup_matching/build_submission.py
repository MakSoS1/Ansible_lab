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
    "ecup_matching/ml/batch_features.py",
    "ecup_matching/ml/model_io.py",
    "ecup_matching/submission/__init__.py",
    "ecup_matching/submission/predict.py",
)


def build_submission(model_path: Path, manifest_path: Path, output_path: Path) -> Path:
    model_path = Path(model_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise ValueError("model file is missing or empty")
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise ValueError("manifest file is missing or empty")

    repo_root = Path(__file__).resolve().parents[1]
    run_source = repo_root / "ecup_matching/submission/run.py"
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
        zf.write(model_path, "model_v1.joblib")
        zf.write(manifest_path, "model_v1_manifest.json")

    if output_path.stat().st_size >= MAX_ARCHIVE_BYTES:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("submission archive exceeds 5 GB limit")

    with zipfile.ZipFile(output_path) as zf:
        names = zf.namelist()
        required = {"metadata.json", "run.py", "model_v1.joblib", "model_v1_manifest.json"}
        if not required.issubset(names):
            raise RuntimeError("submission archive is missing required root files")
        forbidden = [n for n in names if n.endswith(".parquet") or "HF_TOKEN" in n or n.startswith("__MACOSX/")]
        if forbidden:
            raise RuntimeError(f"forbidden files in submission: {forbidden}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = build_submission(args.model, args.manifest, args.output)
    print(f"Built {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
