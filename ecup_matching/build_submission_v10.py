"""Build the minimal v10 tiny-student submission archive.

v10 deliberately breaks from the v8/v9 multi-stage runtime.  The archive may
contain exactly one model checkpoint and only the first-party import closure of
``run_v10``.  Expensive teachers remain training-only evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from ecup_matching.ci.runtime_closure import (
    V10_ENTRYPOINTS,
    copy_runtime_closure,
    missing_from,
)
from ecup_matching.submission.predict_v10 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_LENGTH,
    EXPECTED_BASE_MODEL,
    validate_v10_metadata,
)
from ecup_matching.submission.v10_contract import assert_student_only_archive


MAX_V10_ARCHIVE_BYTES = 150_000_000
PLATFORM_METADATA = {
    "image": "odsai/ecup26-matching-baseline:1.0",
    "entry_point": "python -u run.py",
}
_MODEL_AUX_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "merges.txt",
    "added_tokens.json",
    "generation_config.json",
}
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def _weight_files(model_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file() and path.name.lower().endswith(_WEIGHT_SUFFIXES)
    )


def _assert_safe_source_model(model_dir: Path) -> Path:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory is missing: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise ValueError("model directory is missing config.json")
    weights = _weight_files(model_dir)
    if len(weights) != 1:
        raise ValueError(
            f"v10 source model must contain exactly one weight file, found {len(weights)}"
        )
    for path in model_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"v10 model must not contain symlinks: {path}")
    return weights[0]


def _copy_minimal_model(model_dir: Path, destination: Path) -> list[str]:
    weight = _assert_safe_source_model(model_dir)
    destination.mkdir(parents=True, exist_ok=False)
    selected = {weight.name}
    selected.update(name for name in _MODEL_AUX_FILES if (model_dir / name).is_file())
    copied: list[str] = []
    for name in sorted(selected):
        source = model_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)
            copied.append(name)
    if "config.json" not in copied:
        raise ValueError("v10 minimal model copy lost config.json")
    return copied


def build_model_metadata(
    *,
    strict_oof: float,
    base_model_revision: str,
    max_length: int,
    max_chars: int,
    inference_batch_size: int,
) -> dict[str, object]:
    if not base_model_revision or len(base_model_revision.strip()) < 7:
        raise ValueError("base_model_revision must be an immutable revision identifier")
    payload: dict[str, object] = {
        "version": "v10-tiny-student",
        "base_model": EXPECTED_BASE_MODEL,
        "base_model_revision": base_model_revision.strip(),
        "strict_oof_macro_average_precision": float(strict_oof),
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "max_length": int(max_length),
        "max_chars": int(max_chars),
        "inference_batch_size": int(inference_batch_size),
        "is_production_refit": True,
        "runtime_architecture": "single-small-cross-encoder",
        "heavyweight_teacher_in_submission": False,
    }
    return validate_v10_metadata(payload)


def build_submission_v10(
    *,
    model_dir: Path,
    output_zip: Path,
    strict_oof: float,
    base_model_revision: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_chars: int = DEFAULT_MAX_CHARS,
    inference_batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, object]:
    model_dir = Path(model_dir)
    output_zip = Path(output_zip)
    metadata = build_model_metadata(
        strict_oof=strict_oof,
        base_model_revision=base_model_revision,
        max_length=max_length,
        max_chars=max_chars,
        inference_batch_size=inference_batch_size,
    )
    _assert_safe_source_model(model_dir)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ecup-v10-build-") as raw:
        root = Path(raw) / "submission"
        root.mkdir()
        (root / "metadata.json").write_text(
            json.dumps(PLATFORM_METADATA, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        packaged = copy_runtime_closure(root, V10_ENTRYPOINTS)
        run_source = root / "ecup_matching" / "submission" / "run_v10.py"
        if not run_source.is_file():
            raise RuntimeError("derived v10 runtime closure did not contain run_v10.py")
        shutil.copy2(run_source, root / "run.py")
        (root / "model_v10_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        copied_model_files = _copy_minimal_model(model_dir, root / "model_v10_student")

        gaps = missing_from(root, V10_ENTRYPOINTS)
        if gaps:
            raise RuntimeError(f"v10 runtime closure incomplete before zip: {gaps}")

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())

    contract = assert_student_only_archive(
        output_zip, max_bytes=MAX_V10_ARCHIVE_BYTES
    )
    return {
        "zip": str(output_zip),
        "bytes": int(output_zip.stat().st_size),
        "packaged_modules": packaged,
        "model_files": copied_model_files,
        "model_metadata": metadata,
        "contract": contract,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--strict-oof", type=float, required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--inference-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    report = build_submission_v10(
        model_dir=args.model_dir,
        output_zip=args.output_zip,
        strict_oof=args.strict_oof,
        base_model_revision=args.base_model_revision,
        max_length=args.max_length,
        max_chars=args.max_chars,
        inference_batch_size=args.inference_batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
