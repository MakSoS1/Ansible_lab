from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Callable

PINNED_COMMIT = "bc64ff745a9b2c0e6245e0137654f041fba0c155"
PROFILE = "ecup-local-only-v1"


class HardeningError(RuntimeError):
    """Raised when pinned upstream no longer matches a required hardening anchor."""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text and old not in text:
        return text
    count = text.count(old)
    if count != 1:
        raise HardeningError(f"{label}: expected exactly one upstream anchor, found {count}")
    return text.replace(old, new, 1)


def _function_block(text: str, name: str) -> tuple[int, int, str]:
    pattern = re.compile(
        rf"^def {re.escape(name)}\(.*?(?=^def |^class |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise HardeningError(f"missing function anchor: {name}")
    return match.start(), match.end(), match.group(0)


def _rewrite_function(text: str, name: str, transform: Callable[[str], str]) -> str:
    start, end, block = _function_block(text, name)
    new_block = transform(block)
    if new_block == block:
        return text
    return text[:start] + new_block + text[end:]


def harden_pyproject_text(text: str) -> str:
    return _replace_once(
        text,
        '"mcp>=1.0.0"',
        '"mcp>=1.0.0,<2"',
        "pyproject mcp constraint",
    )


_REDACT_STRUCTURE_HELPER = '''def _redact_structure(value: Any) -> Any:
    """Recursively redact secret-like strings before any metadata is persisted."""
    if isinstance(value, str):
        return _redact_secrets(value)[0]
    if isinstance(value, Mapping):
        return {key: _redact_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_structure(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_structure(item) for item in value]
    return value


'''


def harden_storage_text(text: str) -> str:
    text = _replace_once(
        text,
        'os.getenv("MEMORA_EMBEDDING_MODEL", "openai")',
        'os.getenv("MEMORA_EMBEDDING_MODEL", "tfidf")',
        "storage embedding default",
    )
    text = _replace_once(
        text,
        'os.getenv("MEMORA_LLM_ENABLED", "true")',
        'os.getenv("MEMORA_LLM_ENABLED", "false")',
        "storage LLM default",
    )

    if "def _redact_structure(" not in text:
        anchor = "def _validate_content(content: str) -> str:\n"
        if text.count(anchor) != 1:
            raise HardeningError("storage redact helper: _validate_content anchor mismatch")
        text = text.replace(anchor, _REDACT_STRUCTURE_HELPER + anchor, 1)

    def patch_validate(block: str) -> str:
        if "_redacted_types = _redact_secrets(content)" in block:
            return block
        old = "    return content\n"
        if block.count(old) != 1:
            raise HardeningError("_validate_content return anchor mismatch")
        new = (
            "    content, _redacted_types = _redact_secrets(content)\n"
            "    if _redacted_types:\n"
            "        logger.warning(\"Redacted secret-like value(s) before memory persistence: %s\", \", \".join(_redacted_types))\n"
            "    return content\n"
        )
        return block.replace(old, new, 1)

    text = _rewrite_function(text, "_validate_content", patch_validate)

    def patch_metadata(block: str) -> str:
        old = "_process_metadata_images(dict(metadata), memory_id=memory_id)"
        new = "_process_metadata_images(_redact_structure(dict(metadata)), memory_id=memory_id)"
        if new in block:
            return block
        if block.count(old) != 1:
            raise HardeningError("_prepare_metadata redaction anchor mismatch")
        return block.replace(old, new, 1)

    text = _rewrite_function(text, "_prepare_metadata", patch_metadata)

    def patch_tags(block: str) -> str:
        marker = "stripped, _ = _redact_secrets(stripped)"
        if marker in block:
            return block
        old = "        stripped = tag.strip()\n"
        if block.count(old) != 1:
            raise HardeningError("_validate_tags redaction anchor mismatch")
        new = old + "        stripped, _ = _redact_secrets(stripped)\n"
        return block.replace(old, new, 1)

    text = _rewrite_function(text, "_validate_tags", patch_tags)

    def patch_batch(block: str) -> str:
        old = '        content = str(entry["content"]).strip()\n'
        new = '        content = _validate_content(entry["content"])\n'
        if new in block:
            return block
        if block.count(old) != 1:
            raise HardeningError("add_memories content validation anchor mismatch")
        return block.replace(old, new, 1)

    text = _rewrite_function(text, "add_memories", patch_batch)
    return text


def _append_after_line(text: str, code: str, added: str, label: str) -> str:
    if added in text:
        return text
    pattern = re.compile(rf"^(?P<indent>[ \t]*){re.escape(code)}[ \t]*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise HardeningError(f"{label}: expected exactly one line anchor, found {len(matches)}")
    match = matches[0]
    indent = match.group("indent")
    replacement = f"{indent}{code}\n{indent}{added}"
    return text[: match.start()] + replacement + text[match.end() :]


def harden_backends_text(text: str) -> str:
    anchors = [
        (
            "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
            "os.chmod(self.db_path.parent, 0o700)",
            "local DB directory mode",
        ),
        (
            "conn = sqlite3.connect(self.db_path, check_same_thread=check_same_thread)",
            "os.chmod(self.db_path, 0o600)",
            "local DB file mode",
        ),
        (
            "self.cache_dir.mkdir(parents=True, exist_ok=True)",
            "os.chmod(self.cache_dir, 0o700)",
            "cloud cache root mode",
        ),
        (
            "self.cache_path.parent.mkdir(parents=True, exist_ok=True)",
            "os.chmod(self.cache_path.parent, 0o700)",
            "cloud cache DB directory mode",
        ),
    ]
    for code, added, label in anchors:
        text = _append_after_line(text, code, added, label)
    return text


_DISABLED_GRAPH_HTML = """<!doctype html>
<html><head><meta charset=\"utf-8\"><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\"><title>Memora graph disabled</title></head>
<body><h1>Memora graph disabled</h1><p>This E-CUP hardened profile is local stdio-only. Interactive graph/cloud surfaces are intentionally disabled.</p></body></html>
"""


def harden_tree(source: Path) -> dict[str, object]:
    source = Path(source)
    required = [
        source / "pyproject.toml",
        source / "memora" / "storage.py",
        source / "memora" / "backends.py",
        source / "memora" / "graph" / "index.html",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise HardeningError(f"missing pinned upstream files: {missing}")

    pyproject = source / "pyproject.toml"
    storage = source / "memora" / "storage.py"
    backends = source / "memora" / "backends.py"
    graph_html = source / "memora" / "graph" / "index.html"

    pyproject.write_text(harden_pyproject_text(pyproject.read_text(encoding="utf-8")), encoding="utf-8")
    storage.write_text(harden_storage_text(storage.read_text(encoding="utf-8")), encoding="utf-8")
    backends.write_text(harden_backends_text(backends.read_text(encoding="utf-8")), encoding="utf-8")

    cloud_graph = source / "memora-graph"
    removed = cloud_graph.exists()
    if removed:
        shutil.rmtree(cloud_graph)
    graph_html.write_text(_DISABLED_GRAPH_HTML, encoding="utf-8")

    report = {
        "profile": PROFILE,
        "pinned_commit": PINNED_COMMIT,
        "cloud_graph_removed": removed,
        "graph_ui_disabled": True,
        "mcp_constraint": ">=1,<2",
        "embedding_default": "tfidf",
        "llm_default": False,
    }
    (source / "HARDENED_ECUP_PROFILE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the pinned E-CUP Memora hardening profile")
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    report = harden_tree(args.source)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
