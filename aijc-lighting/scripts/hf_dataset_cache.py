from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT / "data_освещённость.zip"
DEFAULT_REPO_NAME = "aijc-lighting-official-data"
DEFAULT_FILENAME = "data_освещённость.zip"


def dataset_repo_id(owner: str, repo_name: str = DEFAULT_REPO_NAME) -> str:
    owner = str(owner).strip()
    if not owner:
        raise ValueError("Hugging Face owner must not be empty")
    return f"{owner}/{repo_name}"


def resolve_repo_id(token: str, repo_name: str = DEFAULT_REPO_NAME) -> str:
    explicit = os.getenv("AIJC_HF_DATASET_ID", "").strip()
    if explicit:
        return explicit
    owner = HfApi(token=token).whoami()["name"]
    return dataset_repo_id(owner, repo_name)


def write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def restore(token: str, repo_id: str, archive: Path, filename: str) -> bool:
    if archive.exists() and archive.stat().st_size > 0:
        print(f"HF cache restore skipped: archive already exists ({archive.stat().st_size} bytes)")
        return True
    try:
        cached = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=token,
        )
    except (HfHubHTTPError, FileNotFoundError, OSError) as exc:
        print(f"HF cache miss for {repo_id}/{filename}: {type(exc).__name__}: {exc}")
        return False
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, archive)
    print(f"Restored {archive} from private HF dataset {repo_id} ({archive.stat().st_size} bytes)")
    return True


def publish(token: str, repo_id: str, archive: Path, filename: str) -> None:
    if not archive.exists() or archive.stat().st_size == 0:
        raise FileNotFoundError(f"Archive is missing or empty: {archive}")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(archive),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Cache official AIIJC lighting dataset archive",
    )
    info = api.dataset_info(repo_id, files_metadata=True)
    names = {item.rfilename for item in info.siblings}
    if filename not in names:
        raise RuntimeError(f"HF upload verification failed: {filename} is not in {repo_id}")
    print(f"Published and verified private HF dataset cache: {repo_id}/{filename}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["restore", "publish"])
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--repo-name", default=DEFAULT_REPO_NAME)
    parser.add_argument("--state", type=Path, default=PROJECT_ROOT / "data" / "hf_cache_state.json")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is not configured; continuing without Hugging Face cache")
        write_state(args.state, {"enabled": False, "action": args.action})
        return 0

    repo_id = resolve_repo_id(token, args.repo_name)
    if args.action == "restore":
        hit = restore(token, repo_id, args.archive, args.filename)
        write_state(args.state, {
            "enabled": True,
            "action": "restore",
            "repo_id": repo_id,
            "hit": hit,
        })
        return 0

    publish(token, repo_id, args.archive, args.filename)
    write_state(args.state, {
        "enabled": True,
        "action": "publish",
        "repo_id": repo_id,
        "published": True,
        "size_bytes": args.archive.stat().st_size,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
