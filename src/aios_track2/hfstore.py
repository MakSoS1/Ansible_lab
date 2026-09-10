from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    git_sha: str
    github_run_id: str
    dataset_id: str
    seed: int
    simulator_version: str
    deck_sha256: str
    schedule_sha256: str
    status: str
    npv_rub: float | None = None

    def prefix(self) -> str:
        safe_sha = self.git_sha[:12]
        return f"runs/{safe_sha}-{self.github_run_id}/{self.run_id}"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def upload_run_directory(local_dir: Path, manifest: RunManifest, *, token: str) -> None:
    if not token:
        raise ValueError("HF token is required")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("install project with [hf] extras") from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=manifest.dataset_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=manifest.dataset_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        path_in_repo=manifest.prefix(),
        commit_message=f"track2 run {manifest.run_id}",
    )
