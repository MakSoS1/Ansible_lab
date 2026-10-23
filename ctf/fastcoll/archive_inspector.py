#!/usr/bin/env python3
"""Static-only, bounded inspection of an untrusted CTF tarball.

No extraction, execution, imports from archive, subprocesses or network access.
"""
from __future__ import annotations
import hashlib
import io
import os
from pathlib import Path
import sys
import tarfile

MAX_ARCHIVE = 10 * 1024 * 1024
MAX_MEMBERS = 100
MAX_SINGLE_MEMBER = 8 * 1024 * 1024
MAX_OUTPUT_TOTAL = 90_000
MAX_OUTPUT_PER_FILE = 28_000
TEXT_SUFFIXES = {
    ".py", ".js", ".json", ".txt", ".md", ".html", ".css", ".yaml",
    ".yml", ".sh", ".c", ".h", ".cpp", ".hpp", ".go", ".php",
    ".toml", ".conf", ".ini", ".xml", ".sql", ".pl", ".rb",
    ".dockerfile", ".gitignore",
}


def inspect(path: Path) -> None:
    size = path.stat().st_size
    if size > MAX_ARCHIVE:
        raise ValueError(f"archive too large: {size}")
    print(f"ARCHIVE size={size} SHA256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    total_printed = 0
    with tarfile.open(path, mode="r:gz", errorlevel=2) as tar:
        for index, member in enumerate(tar):
            if index >= MAX_MEMBERS:
                print("WARNING: member count cap exceeded; stop")
                break
            name = member.name
            unsafe_path = name.startswith("/") or ".." in Path(name).parts
            kind = (
                "REG" if member.isfile() else
                "DIR" if member.isdir() else
                "SYMLINK" if member.issym() else
                "HARDLINK" if member.islnk() else
                "SPECIAL"
            )
            print(f"MEMBER #{index} name={name!r} type={kind} size={member.size} mode={member.mode:o} unsafe_path={unsafe_path} link={member.linkname!r}")
            if unsafe_path or not member.isfile() or member.size > MAX_SINGLE_MEMBER:
                continue
            suffix = Path(name.lower()).suffix
            text_name = Path(name).name.lower()
            if suffix not in TEXT_SUFFIXES and text_name not in {"dockerfile", "makefile", "readme", ".env.example", "requirements.txt"}:
                continue
            if total_printed >= MAX_OUTPUT_TOTAL:
                print("OUTPUT BUDGET EXCEEDED")
                break
            stream = tar.extractfile(member)
            if stream is None:
                continue
            max_bytes = min(MAX_OUTPUT_PER_FILE, MAX_OUTPUT_TOTAL - total_printed)
            raw = stream.read(max_bytes + 1)
            if b"\0" in raw[:500]:
                print("BINARY CONTENT DETECTED - NOT PRINTING")
                continue
            print(f"=== TEXT DATA {name!r} ({min(len(raw), max_bytes)} bytes sampled) ===")
            for line in raw[:max_bytes].decode("utf-8", errors="replace").splitlines():
                print("DATA " + line.encode("unicode_escape").decode("ascii"))
            if len(raw) > max_bytes:
                print("TRUNCATED")
            print("=== END TEXT DATA ===")
            total_printed += min(len(raw), max_bytes)
    print("Inspection complete: archive contents were NOT executed or extracted.")


if __name__ == "__main__":
    inspect(Path(sys.argv[1]))
