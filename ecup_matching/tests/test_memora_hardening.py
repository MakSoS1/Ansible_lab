import json
from pathlib import Path

from tools.memora_hardened.harden import (
    PINNED_COMMIT,
    harden_backends_text,
    harden_pyproject_text,
    harden_storage_text,
    harden_tree,
)


def test_memora_pin_is_exact_audited_commit():
    assert PINNED_COMMIT == "bc64ff745a9b2c0e6245e0137654f041fba0c155"


def test_pyproject_constrains_mcp_below_v2():
    source = 'dependencies = [\n    "mcp>=1.0.0",\n    "Pillow>=10.4.0",\n]\n'
    hardened = harden_pyproject_text(source)
    assert '"mcp>=1.0.0,<2"' in hardened
    assert '"mcp>=1.0.0",' not in hardened


def test_storage_defaults_are_local_and_redaction_is_centralized():
    source = '''EMBEDDING_MODEL = os.getenv("MEMORA_EMBEDDING_MODEL", "openai")
LLM_ENABLED = os.getenv("MEMORA_LLM_ENABLED", "true").lower() in ("true", "1", "yes")

def _validate_content(content: str) -> str:
    content = content.strip()
    return content

def _prepare_metadata(metadata, memory_id=None):
    if metadata is None:
        return None
    processed = _process_metadata_images(dict(metadata), memory_id=memory_id)
    return _build_metadata_dict(processed)

def _validate_tags(tags):
    if tags is None:
        return []
    validated = []
    for tag in tags:
        stripped = tag.strip()
        validated.append(stripped)
    return validated

def add_memories(conn, entries):
    prepared = []
    for entry in entries:
        content = str(entry["content"]).strip()
        prepared.append(content)
'''
    hardened = harden_storage_text(source)
    assert 'MEMORA_EMBEDDING_MODEL", "tfidf"' in hardened
    assert 'MEMORA_LLM_ENABLED", "false"' in hardened
    assert "content, _redacted_types = _redact_secrets(content)" in hardened
    assert "_redact_structure(dict(metadata))" in hardened
    assert "stripped, _ = _redact_secrets(stripped)" in hardened
    assert 'content = _validate_content(entry["content"])' in hardened


def test_backends_enforce_private_modes():
    source = '''self.db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(self.db_path, check_same_thread=check_same_thread)
self.cache_dir.mkdir(parents=True, exist_ok=True)
self.cache_path.parent.mkdir(parents=True, exist_ok=True)
'''
    hardened = harden_backends_text(source)
    assert "os.chmod(self.db_path.parent, 0o700)" in hardened
    assert "os.chmod(self.db_path, 0o600)" in hardened
    assert "os.chmod(self.cache_dir, 0o700)" in hardened
    assert "os.chmod(self.cache_path.parent, 0o700)" in hardened


def test_harden_tree_removes_cloud_graph_and_disables_packaged_graph(tmp_path: Path):
    (tmp_path / "memora").mkdir()
    (tmp_path / "memora" / "graph").mkdir()
    (tmp_path / "memora" / "graph" / "index.html").write_text("unsafe graph", encoding="utf-8")
    (tmp_path / "memora-graph").mkdir()
    (tmp_path / "memora-graph" / "worker.ts").write_text("unsafe worker", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('"mcp>=1.0.0"', encoding="utf-8")
    (tmp_path / "memora" / "storage.py").write_text(
        'EMBEDDING_MODEL = os.getenv("MEMORA_EMBEDDING_MODEL", "openai")\n'
        'LLM_ENABLED = os.getenv("MEMORA_LLM_ENABLED", "true").lower() in ("true", "1", "yes")\n'
        'def _validate_content(content: str) -> str:\n    return content\n'
        'def _prepare_metadata(metadata, memory_id=None):\n    if metadata is None:\n        return None\n    processed = _process_metadata_images(dict(metadata), memory_id=memory_id)\n    return _build_metadata_dict(processed)\n'
        'def _validate_tags(tags):\n    if tags is None:\n        return []\n    validated = []\n    for tag in tags:\n        stripped = tag.strip()\n        validated.append(stripped)\n    return validated\n'
        'def add_memories(conn, entries):\n    for entry in entries:\n        content = str(entry["content"]).strip()\n',
        encoding="utf-8",
    )
    (tmp_path / "memora" / "backends.py").write_text(
        'import os\nimport sqlite3\n'
        'self.db_path.parent.mkdir(parents=True, exist_ok=True)\n'
        'conn = sqlite3.connect(self.db_path, check_same_thread=check_same_thread)\n'
        'self.cache_dir.mkdir(parents=True, exist_ok=True)\n'
        'self.cache_path.parent.mkdir(parents=True, exist_ok=True)\n',
        encoding="utf-8",
    )

    report = harden_tree(tmp_path)

    assert not (tmp_path / "memora-graph").exists()
    graph_html = (tmp_path / "memora" / "graph" / "index.html").read_text(encoding="utf-8")
    assert "disabled" in graph_html.lower()
    assert report["pinned_commit"] == PINNED_COMMIT
    assert report["cloud_graph_removed"] is True
