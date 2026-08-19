from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StructuredArtifactPaths:
    model: Path
    legacy_runtime: Path


def _valid_legacy_runtime(path: Path) -> bool:
    return path.is_dir() and (path / "__init__.py").is_file() and (path / "ml" / "textnorm.py").is_file()


def _inventory(root: Path, *, limit: int = 80) -> list[str]:
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    if len(files) > limit:
        return [*files[:limit], f"... ({len(files) - limit} more files)"]
    return files


def resolve_structured_artifact(root: Path) -> StructuredArtifactPaths:
    """Resolve the structured model and legacy runtime from extracted Actions artifacts.

    `actions/upload-artifact` preserves a common parent when multiple paths are uploaded,
    so the same logical artifact may be either flat or rooted under `out/` and `legacy/`.
    """

    root = Path(root)
    model_candidates = (
        root / "model_v5_structured.joblib",
        root / "out" / "model_v5_structured.joblib",
    )
    legacy_candidates = (
        root / "legacy_ecup",
        root / "legacy" / "legacy_ecup",
    )

    model = next((path for path in model_candidates if path.is_file()), None)
    legacy_runtime = next((path for path in legacy_candidates if _valid_legacy_runtime(path)), None)
    if model is not None and legacy_runtime is not None:
        return StructuredArtifactPaths(model=model, legacy_runtime=legacy_runtime)

    inventory = _inventory(root)
    raise FileNotFoundError(
        "unable to resolve v5 structured artifact: expected model_v5_structured.joblib "
        "and a legacy_ecup runtime; discovered files=" + repr(inventory)
    )
