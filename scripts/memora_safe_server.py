from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

_UNSAFE_EXACT = {
    "MEMORA_STORAGE_URI",
    "MEMORA_GRAPH_PORT",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLOUDFLARE_API_TOKEN",
    "CHAT_MODEL",
    "R2_PUBLIC_DOMAIN",
    "SENTENCE_TRANSFORMERS_MODEL",
}
_UNSAFE_PREFIXES = ("AWS_", "CLOUDFLARE_", "OPENAI_", "OPENROUTER_", "ANTHROPIC_", "R2_")


def build_safe_env(repo_root: Path, inherited: Mapping[str, str]) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    safe: dict[str, str] = {}
    for key, value in inherited.items():
        if key in _UNSAFE_EXACT or key.startswith(_UNSAFE_PREFIXES):
            continue
        safe[key] = value
    safe.update(
        {
            "MEMORA_DB_PATH": str(repo_root / ".agent-memory" / "memories.db"),
            "MEMORA_EMBEDDING_MODEL": "tfidf",
            "MEMORA_LLM_ENABLED": "false",
            "MEMORA_AUTO_CAPTURE": "false",
        }
    )
    safe.pop("MEMORA_STORAGE_URI", None)
    safe.pop("MEMORA_GRAPH_PORT", None)
    return safe


def _server_path(repo_root: Path) -> Path:
    runtime = repo_root / ".agent-memory" / "runtime" / "venv"
    if os.name == "nt":
        return runtime / "Scripts" / "memora-server.exe"
    return runtime / "bin" / "memora-server"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if len(sys.argv) != 1:
        raise SystemExit("Safe E-CUP Memora launcher accepts no extra arguments; stdio + --no-graph is mandatory")
    server = _server_path(repo_root)
    if not server.is_file():
        raise SystemExit(
            "Hardened Memora runtime not installed. Run: "
            "python tools/memora_hardened/install.py --prefix .agent-memory/runtime"
        )
    memory_dir = repo_root / ".agent-memory"
    if memory_dir.is_symlink():
        raise SystemExit("Refusing symlink .agent-memory directory")
    memory_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(memory_dir, 0o700)
    env = build_safe_env(repo_root, os.environ)
    os.execve(str(server), [str(server), "--no-graph"], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
