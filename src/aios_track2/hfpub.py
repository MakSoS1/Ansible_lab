from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def publish_run(folder: Path, dataset_id: str | None = None) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to publish runs")
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    target = dataset_id or os.environ.get("HF_DATASET_ID") or f"{owner}/aios-track2-runs"
    git_sha = os.environ.get("GITHUB_SHA", "local")[:12]
    run_id = os.environ.get("GITHUB_RUN_ID", "manual")
    dest = f"runs/{git_sha}-{run_id}"
    try:
        files = api.list_repo_files(target, repo_type="dataset")
    except Exception:
        files = []
    prefix = dest.rstrip("/") + "/"
    if any(name.startswith(prefix) for name in files) or dest in files:
        raise RuntimeError(f"run path already exists: {dest}")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=target,
        repo_type="dataset",
        path_in_repo=dest,
    )
    return f"{target}/{dest}"
