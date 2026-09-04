from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .summary_requests import build_training_summary


@dataclass(frozen=True, slots=True)
class SummaryInstallReport:
    summary_path: str
    before_sha256: str
    after_sha256: str
    changed_files: tuple[str, ...]
    unchanged_files: tuple[str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deck_text_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".data", ".inc"}
        )
    )


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def install_training_summary(root: Path) -> SummaryInstallReport:
    """Install training SUMMARY requests while proving the physical deck is immutable."""
    root = root.resolve()
    files = _deck_text_files(root)
    candidates = tuple(path for path in files if "summary" in path.stem.casefold())
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one summary include, found {len(candidates)}")

    summary = candidates[0]
    before = {_relative(root, path): _sha256(path) for path in files}
    before_summary = before[_relative(root, summary)]

    summary.write_text(build_training_summary(), encoding="utf-8", newline="\n")

    after_files = _deck_text_files(root)
    after = {_relative(root, path): _sha256(path) for path in after_files}
    if set(after) != set(before):
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        raise RuntimeError(f"deck file set changed while installing summary: added={added}, removed={removed}")

    changed = tuple(sorted(path for path in before if before[path] != after[path]))
    summary_rel = _relative(root, summary)
    if changed != (summary_rel,):
        raise RuntimeError(f"training telemetry unexpectedly changed physical deck files: {changed}")

    unchanged = tuple(sorted(path for path in before if path not in changed))
    return SummaryInstallReport(
        summary_path=summary_rel,
        before_sha256=before_summary,
        after_sha256=after[summary_rel],
        changed_files=changed,
        unchanged_files=unchanged,
    )
