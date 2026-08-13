from __future__ import annotations

from pathlib import Path
import zipfile


FORBIDDEN_MEMBER_MARKERS = (
    "teacher",
    "contrastive",
    "structured",
    "tfidf",
    "sparse",
    "hgb",
    "histgradient",
    "meta_model",
    "joblib",
)


def assert_student_only_archive(path: Path | str, *, max_bytes: int) -> dict[str, object]:
    archive_path = Path(path)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    size = archive_path.stat().st_size
    if size > int(max_bytes):
        raise ValueError(f"archive size {size} exceeds limit {max_bytes}")

    with zipfile.ZipFile(archive_path) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        bad_zip = archive.testzip()
        if bad_zip is not None:
            raise ValueError(f"corrupt archive member: {bad_zip}")

    lowered = [(name, name.casefold()) for name in names]
    forbidden = sorted(
        name
        for name, low in lowered
        if any(marker in low for marker in FORBIDDEN_MEMBER_MARKERS)
    )
    if forbidden:
        raise ValueError(f"forbidden heavyweight members in v10 archive: {forbidden}")

    weight_files = sorted(
        name for name, low in lowered if low.endswith((".safetensors", ".bin", ".pt", ".pth"))
    )
    if len(weight_files) != 1:
        raise ValueError(
            f"v10 student archive must contain exactly one model weight file, found {len(weight_files)}"
        )

    required_root = {"run.py"}
    missing = sorted(required_root - set(names))
    if missing:
        raise ValueError(f"v10 archive missing required members: {missing}")

    return {
        "bytes": int(size),
        "members": int(len(names)),
        "forbidden_members": forbidden,
        "model_weight_files": int(len(weight_files)),
        "weight_file": weight_files[0],
    }


__all__ = ["assert_student_only_archive", "FORBIDDEN_MEMBER_MARKERS"]
