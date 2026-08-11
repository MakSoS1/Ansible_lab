from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path


METADATA = {
    "image": "odsai/ecup26-matching-baseline:1.0",
    "entry_point": "python -u run.py",
}
MAX_ARCHIVE_BYTES = 5 * 1024**3
SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
RUNTIME_FILES = (
    "ecup_matching/ml/__init__.py",
    "ecup_matching/ml/textnorm.py",
    "ecup_matching/ml/features.py",
    "ecup_matching/ml/category_attrs.py",
    "ecup_matching/ml/features_v2.py",
    "ecup_matching/ml/model_io.py",
    "ecup_matching/ml/reranker_data.py",
    "ecup_matching/submission/__init__.py",
    "ecup_matching/submission/predict_v3.py",
    "ecup_matching/submission/predict_v4.py",
)


def build_submission_v4(
    *,
    structured_model_path: Path,
    structured_manifest_path: Path,
    neural_model_dir: Path,
    neural_manifest_path: Path,
    output_path: Path,
) -> Path:
    structured_model_path = Path(structured_model_path)
    structured_manifest_path = Path(structured_manifest_path)
    neural_model_dir = Path(neural_model_dir)
    neural_manifest_path = Path(neural_manifest_path)
    output_path = Path(output_path)

    for label, path in (
        ("structured model", structured_model_path),
        ("structured manifest", structured_manifest_path),
        ("neural manifest", neural_manifest_path),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    if not neural_model_dir.is_dir():
        raise ValueError(f"neural model directory is missing: {neural_model_dir}")
    for required in ("config.json", "model.safetensors"):
        if not (neural_model_dir / required).is_file():
            raise ValueError(f"neural model is missing {required}")

    structured_manifest = json.loads(
        structured_manifest_path.read_text(encoding="utf-8")
    )
    if structured_manifest.get("version") != "v2-2024-transfer-structured":
        raise ValueError("unexpected structured v2 manifest version")
    if not isinstance(structured_manifest.get("attribute_importance"), dict):
        raise ValueError("structured manifest is missing attribute_importance")

    neural_manifest = json.loads(neural_manifest_path.read_text(encoding="utf-8"))
    if neural_manifest.get("version") != "v4-strong-reranker":
        raise ValueError("unexpected v4 neural manifest version")
    if neural_manifest.get("base_model") != "ai-forever/ruBert-base":
        raise ValueError("unexpected v4 base model")
    revision = str(neural_manifest.get("base_model_revision", ""))
    if SHA_RE.fullmatch(revision) is None:
        raise ValueError("v4 neural manifest must pin an exact base-model revision")
    category_alphas = neural_manifest.get("category_alphas")
    if not isinstance(category_alphas, dict) or not category_alphas:
        raise ValueError("v4 neural manifest is missing category_alphas")
    if not any(float(value) > 0.0 for value in category_alphas.values()):
        raise ValueError("v4 neural manifest routes zero pairs to the reranker")

    repo_root = Path(__file__).resolve().parents[1]
    run_source = repo_root / "ecup_matching/submission/run_v4.py"
    if not run_source.is_file():
        raise FileNotFoundError(run_source)
    for rel in RUNTIME_FILES:
        if not (repo_root / rel).is_file():
            raise FileNotFoundError(repo_root / rel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr(
            "metadata.json", json.dumps(METADATA, ensure_ascii=False, indent=2) + "\n"
        )
        archive.writestr(
            "ecup_matching/__init__.py",
            '"""E-CUP v4 submission runtime package."""\n',
        )
        archive.write(run_source, "run.py")
        for rel in RUNTIME_FILES:
            archive.write(repo_root / rel, rel)
        archive.write(structured_model_path, "model_v2.joblib")
        archive.write(structured_manifest_path, "model_v2_manifest.json")
        archive.write(neural_manifest_path, "model_v4_manifest.json")
        for path in sorted(neural_model_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(neural_model_dir)
                archive.write(path, str(Path("model_v4") / relative))

    if output_path.stat().st_size >= MAX_ARCHIVE_BYTES:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("v4 submission archive exceeds 5 GiB limit")

    with zipfile.ZipFile(output_path) as archive:
        names = archive.namelist()
        required = {
            "metadata.json",
            "run.py",
            "model_v2.joblib",
            "model_v2_manifest.json",
            "model_v4_manifest.json",
            "model_v4/config.json",
            "model_v4/model.safetensors",
            "ecup_matching/submission/predict_v4.py",
            "ecup_matching/submission/predict_v3.py",
            "ecup_matching/ml/reranker_data.py",
        }
        missing = required - set(names)
        if missing:
            raise RuntimeError(
                f"v4 submission archive is missing files: {sorted(missing)}"
            )
        forbidden_suffixes = (".parquet", ".db", ".pem", ".b64", ".zip")
        suspicious_terms = (
            "hf_token",
            "api_token",
            "access_token",
            "secret",
            "password",
            "credential",
        )
        forbidden = [
            name
            for name in names
            if name.endswith(forbidden_suffixes)
            or name.startswith("__MACOSX/")
            or any(term in name.lower() for term in suspicious_terms)
        ]
        if forbidden:
            raise RuntimeError(f"forbidden files in v4 submission: {forbidden}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-model", required=True, type=Path)
    parser.add_argument("--structured-manifest", required=True, type=Path)
    parser.add_argument("--neural-model-dir", required=True, type=Path)
    parser.add_argument("--neural-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = build_submission_v4(
        structured_model_path=args.structured_model,
        structured_manifest_path=args.structured_manifest,
        neural_model_dir=args.neural_model_dir,
        neural_manifest_path=args.neural_manifest,
        output_path=args.output,
    )
    print(f"Built {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
