from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from scripts.memory_common import (
    HF_REPO_ID,
    MEMORA_PINNED_COMMIT,
    ensure_private_dir,
    ensure_private_file,
    sha256_file,
    sqlite_integrity,
)

REMOTE_MANIFEST = "agent-memory/latest/manifest.json"
REMOTE_DB = "agent-memory/latest/memories.db"


def bootstrap(repo_root: Path, *, allow_empty: bool = False) -> dict[str, object]:
    root = Path(repo_root).resolve()
    memory_dir = ensure_private_dir(root / ".agent-memory")
    destination = memory_dir / "memories.db"
    if destination.is_symlink():
        raise RuntimeError(f"refusing symlink memory database: {destination}")

    token = os.getenv("HF_TOKEN")
    if not token:
        if allow_empty:
            return {"restored": False, "reason": "HF_TOKEN unavailable", "path": str(destination)}
        raise RuntimeError("HF_TOKEN is required to restore private agent memory")

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    with tempfile.TemporaryDirectory(prefix="ecup-memory-bootstrap-") as tmp:
        tmpdir = Path(tmp)
        try:
            manifest_path = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    filename=REMOTE_MANIFEST,
                    token=token,
                    local_dir=tmpdir,
                )
            )
        except EntryNotFoundError:
            if allow_empty:
                return {"restored": False, "reason": "no remote checkpoint yet", "path": str(destination)}
            raise RuntimeError("private HF has no agent-memory/latest checkpoint")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("memora_upstream_commit") != MEMORA_PINNED_COMMIT:
            raise RuntimeError("remote memory checkpoint uses a different Memora pin")
        expected_sha = manifest.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise RuntimeError("remote memory manifest has invalid SHA-256")

        downloaded = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=REMOTE_DB,
                token=token,
                local_dir=tmpdir,
            )
        )
        if sha256_file(downloaded) != expected_sha:
            raise RuntimeError("remote memory database SHA-256 mismatch")

        staging = memory_dir / ".memories.db.new"
        if staging.exists() or staging.is_symlink():
            staging.unlink()
        shutil.copyfile(downloaded, staging)
        ensure_private_file(staging)
        sqlite_integrity(staging)
        os.replace(staging, destination)
        ensure_private_file(destination)
        sqlite_integrity(destination)

    print(f"Restored private agent memory checkpoint to {destination}")
    return {
        "restored": True,
        "path": str(destination),
        "sha256": expected_sha,
        "iteration": manifest.get("iteration"),
        "git_commit": manifest.get("git_commit"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore E-CUP Memora DB from private Hugging Face")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-empty", action="store_true", help="permit initialization when no remote checkpoint exists")
    args = parser.parse_args()
    result = bootstrap(args.repo_root, allow_empty=args.allow_empty)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
