from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


SOURCE_KEEPER_SHA256 = "925456cde1e47c50dc0141ce64bed5ef00d9f574152f285869ebea2db6935782"
SOURCE_KEEPER_BYTES = 1_251_659_961
TARGET_MAX_BYTES = 700 * 1024**2
CONVERT_MODEL_FILES = (
    "model_v5_contrastive/model.safetensors",
    "model_v5_teacher/model.safetensors",
)
REQUIRED_MEMBERS = (
    "run.py",
    "metadata.json",
    "model_v5_structured.joblib",
    "model_v5_contrastive/config.json",
    "model_v5_contrastive/model.safetensors",
    "model_v5_teacher/config.json",
    "model_v5_teacher/model.safetensors",
    "model_v6_category_shrunk.json",
    "model_v6_hgb_meta.joblib",
    "model_v6_gate_metadata.json",
    "v9_metadata.json",
    "ecup_matching/submission/predict_v6.py",
    "ecup_matching/ml/v8_submission_graph.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(source_zip: Path, destination: Path) -> list[str]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    with zipfile.ZipFile(source_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError("source ZIP failed CRC validation")
        seen: set[str] = set()
        for info in archive.infolist():
            name = info.filename
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"unsafe ZIP member: {name}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink ZIP member is forbidden: {name}")
            if name in seen:
                raise ValueError(f"duplicate ZIP member: {name}")
            seen.add(name)
            names.append(name)
            target = destination.joinpath(*path.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 4 * 1024 * 1024)
    return names


def convert_safetensors_fp16(source: Path, destination: Path) -> dict[str, object]:
    """Store wide floating tensors in fp16 while preserving non-floating tensors.

    This is a storage transformation only. Runtime equivalence is deliberately
    verified separately in the organizer image before the package can become a
    submission keeper.
    """

    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file

    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"safetensors source must be a regular file: {source}")

    with safe_open(str(source), framework="np") as handle:
        metadata = dict(handle.metadata() or {})
    tensors = load_file(str(source))
    converted: dict[str, np.ndarray] = {}
    converted_names: list[str] = []
    preserved_names: list[str] = []
    source_tensor_bytes = 0
    output_tensor_bytes = 0
    for name in sorted(tensors):
        array = np.asarray(tensors[name])
        source_tensor_bytes += int(array.nbytes)
        if np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize > 2:
            out = array.astype(np.float16)
            converted_names.append(name)
        else:
            out = array
            preserved_names.append(name)
        converted[name] = out
        output_tensor_bytes += int(out.nbytes)

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(destination), metadata=metadata)
    del converted, tensors

    return {
        "source_bytes": int(source.stat().st_size),
        "output_bytes": int(destination.stat().st_size),
        "source_tensor_bytes": source_tensor_bytes,
        "output_tensor_bytes": output_tensor_bytes,
        "converted_tensors": converted_names,
        "preserved_tensors": preserved_names,
        "metadata": metadata,
    }


def _remove_dead_bytecode(root: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.as_posix(), reverse=True):
        if path.is_file() and (path.suffix == ".pyc" or "__pycache__" in path.parts):
            removed.append(path.relative_to(root).as_posix())
            path.unlink()
    for directory in sorted(
        [p for p in root.rglob("__pycache__") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        shutil.rmtree(directory, ignore_errors=True)
    return sorted(removed)


def _assert_required_tree(root: Path) -> None:
    missing = [name for name in REQUIRED_MEMBERS if not (root / name).is_file()]
    if missing:
        raise ValueError(f"compact source is missing required runtime members: {missing}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"compact tree contains symlink: {path}")


def _write_zip(root: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(output_zip) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("compact ZIP failed CRC validation")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("compact ZIP contains duplicate member names")


def compact_v9_submission(
    *,
    source_zip: Path,
    output_zip: Path,
    enforce_source_keeper: bool = True,
    max_output_bytes: int | None = None,
) -> dict[str, object]:
    source_zip = Path(source_zip).resolve(strict=True)
    output_zip = Path(output_zip)
    source_sha = _sha256(source_zip)
    source_bytes = int(source_zip.stat().st_size)
    if enforce_source_keeper:
        if source_sha != SOURCE_KEEPER_SHA256:
            raise ValueError(f"wrong source keeper SHA-256: {source_sha}")
        if source_bytes != SOURCE_KEEPER_BYTES:
            raise ValueError(f"wrong source keeper size: {source_bytes}")

    with tempfile.TemporaryDirectory(prefix="ecup-v9-compact-") as raw:
        root = Path(raw) / "submission"
        original_members = _safe_extract(source_zip, root)
        _assert_required_tree(root)
        removed = _remove_dead_bytecode(root)

        conversions: dict[str, dict[str, object]] = {}
        for relative in CONVERT_MODEL_FILES:
            source_model = root / relative
            temp_model = source_model.with_suffix(".compact.safetensors")
            report = convert_safetensors_fp16(source_model, temp_model)
            temp_model.replace(source_model)
            report["output_bytes"] = int(source_model.stat().st_size)
            report["output_sha256"] = _sha256(source_model)
            conversions[relative] = report

        _assert_required_tree(root)
        inherited = json.loads((root / "v9_metadata.json").read_text(encoding="utf-8"))
        compact_meta = {
            "version": "v9-compact-fp16-storage",
            "source_archive": source_zip.name,
            "source_archive_sha256": source_sha,
            "source_archive_bytes": source_bytes,
            "source_v9_version": inherited.get("version"),
            "model_storage": "float tensors wider than 16-bit downcast to IEEE fp16; integer tensors preserved",
            "converted_model_files": list(CONVERT_MODEL_FILES),
            "removed_members": removed,
            "validation": inherited.get("validation", {}),
            "graph": inherited.get("graph", {}),
            "sealed_gold_opened": False,
            "gold_rows_scored": 0,
            "leaderboard_score": None,
            "leaderboard_score_claimed": False,
            "runtime_equivalence_required": True,
            "end_to_end_runtime_gate_required": True,
        }
        (root / "v9_compact_metadata.json").write_text(
            json.dumps(compact_meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        _write_zip(root, output_zip)

    output_bytes = int(output_zip.stat().st_size)
    effective_limit = max_output_bytes
    if effective_limit is None and enforce_source_keeper:
        effective_limit = TARGET_MAX_BYTES
    if effective_limit is not None and output_bytes > int(effective_limit):
        output_zip.unlink(missing_ok=True)
        raise ValueError(
            f"compact archive is {output_bytes} bytes, over limit {int(effective_limit)}"
        )

    return {
        "version": "v9-compact-fp16-storage",
        "source_archive_sha256": source_sha,
        "source_archive_bytes": source_bytes,
        "output": str(output_zip),
        "output_archive_sha256": _sha256(output_zip),
        "output_archive_bytes": output_bytes,
        "size_ratio": output_bytes / source_bytes,
        "bytes_saved": source_bytes - output_bytes,
        "removed_members": removed,
        "converted_model_files": sorted(conversions),
        "conversion_reports": conversions,
        "source_member_count": len(original_members),
        "target_max_bytes": effective_limit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--allow-nonkeeper-source", action="store_true")
    parser.add_argument("--max-output-bytes", type=int, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()
    report = compact_v9_submission(
        source_zip=args.source_zip,
        output_zip=args.output_zip,
        enforce_source_keeper=not args.allow_nonkeeper_source,
        max_output_bytes=args.max_output_bytes,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
