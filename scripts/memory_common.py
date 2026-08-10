from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import stat
from pathlib import Path

HF_REPO_ID = "Maksim123321/e-cup-2026-matching-private"
MEMORY_SCHEMA_VERSION = 1
MEMORA_PINNED_COMMIT = "bc64ff745a9b2c0e6245e0137654f041fba0c155"

SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "Hugging Face token"),
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"), "OpenAI-style API key"),
    (re.compile(r"sk-or-[A-Za-z0-9_-]{20,}"), "OpenRouter API key"),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "Anthropic API key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"), "Private key"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE), "Bearer token"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "GitHub PAT"),
    (re.compile(r"gho_[A-Za-z0-9]{36}"), "GitHub OAuth token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{22,}"), "GitHub fine-grained PAT"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
]

_ASSIGNMENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bpassword\s*[:=]\s*([^\s,;]+)", re.IGNORECASE), "Password in plaintext"),
    (re.compile(r"\bsecret\s*[:=]\s*([^\s,;]+)", re.IGNORECASE), "Secret in plaintext"),
]


def _looks_like_documentation_placeholder(raw_value: str) -> bool:
    value = raw_value.strip().strip("`'\"")
    if not value:
        return True
    if value.upper() in {"[REDACTED]", "REDACTED", "***", "NONE", "NULL"}:
        return True
    if value.startswith("<") and value.endswith(">"):
        return True
    if value.startswith("${") and value.endswith("}"):
        return True
    if value.startswith("$") and re.fullmatch(r"\$[A-Z][A-Z0-9_]{2,}", value):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", value):
        return True
    return False


def scan_text_for_secrets(text: str) -> list[str]:
    findings: list[str] = []
    for pattern, label in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(label)
    for pattern, label in _ASSIGNMENT_PATTERNS:
        for match in pattern.finditer(text):
            if not _looks_like_documentation_placeholder(match.group(1)):
                findings.append(label)
                break
    return sorted(set(findings))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_private_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists() and path.is_symlink():
        raise RuntimeError(f"refusing symlink directory: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    return path


def ensure_private_file(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink file: {path}")
    if not path.is_file():
        raise RuntimeError(f"expected regular file: {path}")
    os.chmod(path, 0o600)
    return path


def assert_private_modes(directory: Path, db_path: Path) -> None:
    dir_mode = stat.S_IMODE(Path(directory).stat().st_mode)
    db_mode = stat.S_IMODE(Path(db_path).stat().st_mode)
    if dir_mode != 0o700:
        raise RuntimeError(f"memory directory mode {oct(dir_mode)} != 0o700")
    if db_mode != 0o600:
        raise RuntimeError(f"memory database mode {oct(db_mode)} != 0o600")


def sqlite_integrity(path: Path) -> None:
    path = ensure_private_file(path)
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        if not row or row[0] != "ok":
            raise RuntimeError(f"SQLite integrity_check failed: {row}")
    finally:
        conn.close()


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def scan_sqlite_for_secrets(path: Path) -> list[str]:
    """Scan all SQLite TEXT columns; returns human-readable table/column findings."""
    path = ensure_private_file(path)
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    findings: list[str] = []
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
            text_cols = [row[1] for row in cols if str(row[2] or "").upper().startswith("TEXT")]
            if not text_cols:
                continue
            select = ", ".join(_quote_identifier(col) for col in text_cols)
            for row in conn.execute(f"SELECT {select} FROM {_quote_identifier(table)}"):
                for column, value in zip(text_cols, row):
                    if not isinstance(value, str) or not value:
                        continue
                    for label in scan_text_for_secrets(value):
                        findings.append(f"{table}.{column}: {label}")
    finally:
        conn.close()
    return sorted(set(findings))
