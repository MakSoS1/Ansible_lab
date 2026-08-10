from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path

from tools.memora_hardened.harden import PINNED_COMMIT


def _venv_python(prefix: Path) -> Path:
    if os.name == "nt":
        return prefix / "venv" / "Scripts" / "python.exe"
    return prefix / "venv" / "bin" / "python"


def verify(prefix: Path) -> dict[str, object]:
    prefix = Path(prefix).resolve()
    manifest = json.loads((prefix / "runtime-manifest.json").read_text(encoding="utf-8"))
    if manifest["upstream_commit"] != PINNED_COMMIT:
        raise RuntimeError("runtime manifest pin mismatch")
    if int(str(manifest["mcp_version"]).split(".", 1)[0]) >= 2:
        raise RuntimeError("runtime has unsupported MCP major")

    source = prefix / "src"
    if (source / "memora-graph").exists():
        raise RuntimeError("cloud graph tree survived hardening")
    graph_html = (source / "memora" / "graph" / "index.html").read_text(encoding="utf-8")
    if "graph disabled" not in graph_html.lower():
        raise RuntimeError("packaged graph UI was not disabled")

    python = _venv_python(prefix)
    with tempfile.TemporaryDirectory(prefix="memora-verify-") as tmp:
        root = Path(tmp)
        db_dir = root / "private"
        db_path = db_dir / "memories.db"
        env = dict(os.environ)
        env.update(
            {
                "MEMORA_DB_PATH": str(db_path),
                "MEMORA_EMBEDDING_MODEL": "tfidf",
                "MEMORA_LLM_ENABLED": "false",
                "MEMORA_AUTO_CAPTURE": "false",
            }
        )
        env.pop("MEMORA_STORAGE_URI", None)
        code = r'''
import json
import memora.storage as storage
with storage.connect() as conn:
    storage.ensure_schema(conn)
    created = storage.add_memory(
        conn,
        content="verification password=NeverStoreThis123",
        metadata={"note": "secret=DoNotPersist456"},
        tags=["ghp_" + "A" * 36],
    )
    fetched = storage.get_memory(conn, created["id"])
    print(json.dumps({
        "embedding_model": storage.EMBEDDING_MODEL,
        "llm_enabled": storage.LLM_ENABLED,
        "content": fetched["content"],
        "metadata": fetched["metadata"],
        "tags": fetched["tags"],
    }, ensure_ascii=False))
'''
        output = subprocess.check_output([str(python), "-c", code], env=env, text=True)
        result = json.loads(output.strip().splitlines()[-1])
        if result["embedding_model"] != "tfidf" or result["llm_enabled"] is not False:
            raise RuntimeError(f"unsafe effective defaults: {result}")
        serialized = json.dumps(result, ensure_ascii=False)
        for secret in ("NeverStoreThis123", "DoNotPersist456", "ghp_" + "A" * 36):
            if secret in serialized:
                raise RuntimeError("secret redaction verification failed")
        if "[REDACTED]" not in serialized:
            raise RuntimeError("expected redaction marker missing")

        dir_mode = stat.S_IMODE(db_dir.stat().st_mode)
        db_mode = stat.S_IMODE(db_path.stat().st_mode)
        if dir_mode != 0o700:
            raise RuntimeError(f"DB directory mode is {oct(dir_mode)}, expected 0o700")
        if db_mode != 0o600:
            raise RuntimeError(f"DB mode is {oct(db_mode)}, expected 0o600")

    return {
        "profile": manifest["profile"],
        "upstream_commit": manifest["upstream_commit"],
        "mcp_version": manifest["mcp_version"],
        "graph_removed": True,
        "redaction_verified": True,
        "permissions_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify effective hardened Memora runtime behavior")
    parser.add_argument("--prefix", type=Path, default=Path(".agent-memory/runtime"))
    args = parser.parse_args()
    print(json.dumps(verify(args.prefix), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
