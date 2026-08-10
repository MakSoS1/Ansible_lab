from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.memory_common import ensure_private_dir, ensure_private_file, scan_text_for_secrets


_BASE_FILES = [
    "AGENTS.md",
    "docs/agent-memory/PROJECT_STATE.md",
    "docs/agent-memory/EXPERIMENT_INDEX.md",
    "docs/agent-memory/DECISIONS.md",
    "docs/agent-memory/SECURITY.md",
    "docs/agent-memory/ITERATION_PROTOCOL.md",
    "ecup_matching/SOLUTION_RESEARCH.md",
    "ecup_matching/BASELINE_CONTRACT.md",
]


def _configure_local_only(root: Path) -> Path:
    memory_dir = ensure_private_dir(root / ".agent-memory")
    db_path = memory_dir / "memories.db"
    for key in list(os.environ):
        if key == "MEMORA_STORAGE_URI" or key.startswith(("OPENAI_", "OPENROUTER_", "ANTHROPIC_", "AWS_", "CLOUDFLARE_", "R2_")):
            os.environ.pop(key, None)
    os.environ["MEMORA_DB_PATH"] = str(db_path)
    os.environ["MEMORA_EMBEDDING_MODEL"] = "tfidf"
    os.environ["MEMORA_LLM_ENABLED"] = "false"
    os.environ["MEMORA_AUTO_CAPTURE"] = "false"
    os.environ["MEMORA_ALLOW_ANY_TAG"] = "1"
    return db_path


def canonical_sources(root: Path) -> list[Path]:
    root = Path(root).resolve()
    paths: set[Path] = set()
    for rel in _BASE_FILES:
        path = root / rel
        if path.is_file():
            paths.add(path)
    for pattern in (
        "docs/superpowers/specs/*.md",
        "docs/superpowers/plans/*.md",
        "ecup_matching/experiments/v*/PLAN.md",
        "ecup_matching/experiments/v*/RESULTS.md",
    ):
        for path in root.glob(pattern):
            if path.is_file():
                paths.add(path)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def _tags_for(rel: str) -> list[str]:
    tags = ["ecup-matching/memory", "ecup-matching/source-backed"]
    if "/experiments/" in rel:
        parts = Path(rel).parts
        try:
            idx = parts.index("experiments")
            version = parts[idx + 1]
            tags.append(f"ecup-matching/experiment/{version}")
        except (ValueError, IndexError):
            pass
    if rel.endswith("PROJECT_STATE.md"):
        tags.append("ecup-matching/project-state")
    if rel.endswith("DECISIONS.md"):
        tags.append("ecup-matching/decisions")
    if rel.endswith("SECURITY.md"):
        tags.append("ecup-matching/security")
    return tags


def ingest(repo_root: Path) -> dict[str, int]:
    root = Path(repo_root).resolve()
    db_path = _configure_local_only(root)

    # Import only after local-only environment is fixed because Memora selects
    # its storage/embedding backend at module import time.
    import memora.storage as storage

    created = 0
    updated = 0
    deduplicated = 0
    sources = canonical_sources(root)
    with storage.connect() as conn:
        storage.ensure_schema(conn)
        for path in sources:
            rel = path.relative_to(root).as_posix()
            content = path.read_text(encoding="utf-8")
            findings = scan_text_for_secrets(content)
            if findings:
                raise RuntimeError(f"refusing to ingest probable secret in {rel}: {', '.join(findings)}")
            source_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            metadata = {
                "project": "ecup-matching",
                "source_path": rel,
                "source_sha256": source_sha,
                "source_backed": True,
            }
            existing = storage.list_memories(
                conn,
                metadata_filters={"source_path": rel},
                limit=-1,
            )
            tags = _tags_for(rel)
            if existing:
                keeper = existing[0]
                storage.update_memory(
                    conn,
                    keeper["id"],
                    content=content,
                    metadata=metadata,
                    tags=tags,
                )
                updated += 1
                for duplicate in existing[1:]:
                    storage.delete_memory(conn, duplicate["id"])
                    deduplicated += 1
            else:
                storage.add_memory(conn, content=content, metadata=metadata, tags=tags)
                created += 1

    ensure_private_file(db_path)
    print(f"Memora source ingest: {created} created, {updated} updated, {deduplicated} duplicates removed")
    return {"sources": len(sources), "created": created, "updated": updated, "deduplicated": deduplicated}


def main() -> int:
    parser = argparse.ArgumentParser(description="Upsert canonical E-CUP documents into local hardened Memora")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    ingest(args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
