"""Static import closure of offline submission entrypoints.

The submission runs offline from an extracted archive with no repository
checkout, so any first-party module that is imported but not packaged fails at
inference time instead of at build time. Deriving the file list from the real
import graph — rather than maintaining it by hand — also prevents the archive
from silently shipping a stale copy of a module the current iteration changed.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "ecup_matching"
ENTRYPOINTS: tuple[str, ...] = ("ecup_matching.submission.run_v6",)
V7_ENTRYPOINTS: tuple[str, ...] = ("ecup_matching.submission.run_v7",)
V10_ENTRYPOINTS: tuple[str, ...] = ("ecup_matching.submission.run_v10",)

# Resolved at runtime from the pinned ``legacy_ecup`` tree that the verified v5
# base archive ships at its own top level, not through ``ecup_matching``.
EXTERNAL_RUNTIME_PACKAGES: tuple[str, ...] = ("legacy_ecup",)

# ``ecup_matching`` is a namespace package in the repository but must be a real
# package inside the archive, so these markers are materialized even when the
# repository has no file for them. Mirrors build_submission_v5._PACKAGE_MARKERS.
PACKAGE_MARKERS: tuple[str, ...] = (
    "ecup_matching/__init__.py",
    "ecup_matching/ml/__init__.py",
    "ecup_matching/submission/__init__.py",
)


def module_path(module: str) -> Path | None:
    relative = Path(*module.split("."))
    for candidate in (
        REPO_ROOT / relative.with_suffix(".py"),
        REPO_ROOT / relative / "__init__.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def _first_party_imports(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module.rsplit(".", 1)[0] if "." in module else module
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == PACKAGE:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                prefix = ".".join(parts[: len(parts) - node.level + 1])
                target = f"{prefix}.{node.module}" if node.module else prefix
            else:
                target = node.module or ""
            if target.split(".")[0] != PACKAGE:
                continue
            if module_path(target) is not None:
                found.add(target)
            for alias in node.names:
                submodule = f"{target}.{alias.name}"
                if module_path(submodule) is not None:
                    found.add(submodule)
    return found


def runtime_modules(entrypoints: tuple[str, ...] = ENTRYPOINTS) -> list[str]:
    """Dotted names of every first-party module reachable from the entrypoints."""
    seen: set[str] = set()
    queue = list(entrypoints)
    while queue:
        module = queue.pop()
        if module in seen or module_path(module) is None:
            continue
        seen.add(module)
        queue.extend(_first_party_imports(module_path(module), module))
    return sorted(seen)


def runtime_import_closure(
    entrypoints: tuple[str, ...] = ENTRYPOINTS,
) -> list[str]:
    """Archive-relative source paths that the submission must contain."""
    paths: set[str] = set(PACKAGE_MARKERS)
    for module in runtime_modules(entrypoints):
        path = module_path(module)
        assert path is not None
        paths.add(str(path.relative_to(REPO_ROOT)))
        parts = module.split(".")
        for depth in range(1, len(parts)):
            marker = REPO_ROOT.joinpath(*parts[:depth], "__init__.py")
            if marker.is_file():
                paths.add(str(marker.relative_to(REPO_ROOT)))
    return sorted(paths)


def copy_runtime_closure(
    destination: Path,
    entrypoints: tuple[str, ...] = ENTRYPOINTS,
) -> list[str]:
    """Copy every runtime module into an extracted submission tree."""
    import shutil

    copied: list[str] = []
    for relative in runtime_import_closure(entrypoints):
        source = REPO_ROOT / relative
        target = Path(destination) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, target)
        elif relative in PACKAGE_MARKERS:
            target.write_text("", encoding="utf-8")
        else:
            raise FileNotFoundError(f"runtime module has no source file: {relative}")
        copied.append(relative)
    return copied


def missing_from(
    destination: Path,
    entrypoints: tuple[str, ...] = ENTRYPOINTS,
) -> list[str]:
    root = Path(destination)
    return [rel for rel in runtime_import_closure(entrypoints) if not (root / rel).is_file()]


def entrypoints_for_iteration(iteration: str) -> tuple[str, ...]:
    if iteration == "v7":
        return V7_ENTRYPOINTS
    if iteration == "v10":
        return V10_ENTRYPOINTS
    if iteration == "v6":
        return ENTRYPOINTS
    raise ValueError(f"unsupported iteration: {iteration}")


if __name__ == "__main__":  # pragma: no cover - CI entrypoint
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-into", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--iteration", choices=("v6", "v7", "v10"), default="v6")
    args = parser.parse_args()
    entrypoints = entrypoints_for_iteration(args.iteration)
    if args.copy_into is not None:
        for name in copy_runtime_closure(args.copy_into, entrypoints):
            print(f"packaged {name}")
    if args.verify is not None:
        gaps = missing_from(args.verify, entrypoints)
        if gaps:
            raise SystemExit(f"submission archive is missing runtime modules: {gaps}")
        print(
            f"runtime import closure verified for {args.iteration}: "
            f"{len(runtime_import_closure(entrypoints))} modules"
        )
