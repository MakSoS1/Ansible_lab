from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.memory_common import (
    HF_REPO_ID,
    MEMORY_SCHEMA_VERSION,
    MEMORA_PINNED_COMMIT,
    ensure_private_file,
    scan_sqlite_for_secrets,
    sha256_file,
    sqlite_integrity,
)
from scripts.memory_policy import validate_repository


def _git_commit(root: Path) -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _sqlite_backup(source: Path, destination: Path) -> None:
    source = ensure_private_file(source)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(destination)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    os.chmod(destination, 0o600)


def checkpoint(repo_root: Path, iteration: str) -> dict[str, object]:
    root = Path(repo_root).resolve()
    policy_errors = validate_repository(root)
    if policy_errors:
        raise RuntimeError("documentation policy failed: " + " | ".join(policy_errors))

    source = root / ".agent-memory" / "memories.db"
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"refusing non-regular memory database: {source}")
    ensure_private_file(source)
    sqlite_integrity(source)

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for private memory checkpoint")

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    git_commit = _git_commit(root)
    checkpoint_id = f"{timestamp}-{iteration}-{git_commit[:8]}"

    with tempfile.TemporaryDirectory(prefix="ecup-memory-checkpoint-") as tmp:
        tmpdir = Path(tmp)
        backup = tmpdir / "memories.db"
        _sqlite_backup(source, backup)
        ensure_private_file(backup)
        sqlite_integrity(backup)
        findings = scan_sqlite_for_secrets(backup)
        if findings:
            preview = "; ".join(findings[:20])
            raise RuntimeError(f"secret scan blocked memory checkpoint: {preview}")

        digest = sha256_file(backup)
        manifest = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "memora_upstream_commit": MEMORA_PINNED_COMMIT,
            "profile": "ecup-local-only-v1",
            "iteration": iteration,
            "git_commit": git_commit,
            "created_at": now.isoformat(),
            "sha256": digest,
            "size_bytes": backup.stat().st_size,
            "checkpoint_id": checkpoint_id,
        }
        manifest_path = tmpdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(manifest_path, 0o600)

        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

        checkpoint_prefix = f"agent-memory/checkpoints/{checkpoint_id}"
        remote_paths = [
            f"{checkpoint_prefix}/memories.db",
            f"{checkpoint_prefix}/manifest.json",
            "agent-memory/latest/memories.db",
            "agent-memory/latest/manifest.json",
        ]
        operations = [
            CommitOperationAdd(path_in_repo=remote_paths[0], path_or_fileobj=str(backup)),
            CommitOperationAdd(path_in_repo=remote_paths[1], path_or_fileobj=str(manifest_path)),
            CommitOperationAdd(path_in_repo=remote_paths[2], path_or_fileobj=str(backup)),
            CommitOperationAdd(path_in_repo=remote_paths[3], path_or_fileobj=str(manifest_path)),
        ]
        api = HfApi(token=token)
        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Checkpoint E-CUP agent memory {checkpoint_id}",
        )

        visible = set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"))
        missing = [path for path in remote_paths if path not in visible]
        if missing:
            raise RuntimeError(f"HF checkpoint verification missing files: {missing}")

        with tempfile.TemporaryDirectory(prefix="ecup-memory-verify-") as verify_tmp:
            downloaded_manifest = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    filename="agent-memory/latest/manifest.json",
                    token=token,
                    local_dir=verify_tmp,
                    force_download=True,
                )
            )
            remote_manifest = json.loads(downloaded_manifest.read_text(encoding="utf-8"))
            if remote_manifest.get("sha256") != digest or remote_manifest.get("checkpoint_id") != checkpoint_id:
                raise RuntimeError("HF latest manifest does not match uploaded checkpoint")

    print(f"Verified private HF memory checkpoint: {checkpoint_id}")
    return {"checkpoint_id": checkpoint_id, "sha256": digest, "remote_paths": remote_paths}


def main() -> int:
    parser = argparse.ArgumentParser(description="Securely checkpoint E-CUP Memora DB to private HF")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--iteration", required=True)
    args = parser.parse_args()
    result = checkpoint(args.repo_root, args.iteration)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
