from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi


DEFAULT_REPO_ID = "Maksim123321/e-cup-2026-matching-private"
CHUNK_SIZE = 8 * 1024 * 1024
HTTP_TIMEOUT = (15, 180)


@dataclass(frozen=True)
class SourceFile:
    name: str
    url: str


SOURCES: tuple[SourceFile, ...] = (
    SourceFile(
        "matches.parquet",
        "https://storage.yandexcloud.net/ozon-ecup-2026/Matching/matches.parquet",
    ),
    SourceFile(
        "matches_llm.parquet",
        "https://storage.yandexcloud.net/ozon-ecup-2026/Matching/matches_llm.parquet",
    ),
    SourceFile(
        "items.parquet",
        "https://storage.yandexcloud.net/ozon-ecup-2026/Matching/items.parquet",
    ),
    SourceFile(
        "items_human.parquet",
        "https://storage.yandexcloud.net/ozon-ecup-2026/Matching/items_human.parquet",
    ),
)


def download_file(
    source: SourceFile,
    destination: Path,
    session: requests.Session,
) -> int:
    """Download one source file with bounded memory usage and return its size."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with session.get(source.url, stream=True, timeout=HTTP_TIMEOUT) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                total += len(chunk)

    if total == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Download of {source.name} produced zero bytes")

    return total


def mirror_files(
    repo_id: str,
    token: str,
    workdir: Path,
    sources: tuple[SourceFile, ...] = SOURCES,
    api=None,
    session=None,
) -> list[str]:
    """Mirror competition files into a private Hugging Face dataset repository."""
    if not token or not token.strip():
        raise ValueError("HF_TOKEN is required to create and upload the private dataset")

    workdir.mkdir(parents=True, exist_ok=True)
    api = api or HfApi(token=token)
    session = session or requests.Session()

    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    verified: list[str] = []

    for source in sources:
        local_path = workdir / source.name
        try:
            size = download_file(source, local_path, session)
            print(f"Downloaded {source.name}: {size} bytes")

            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=source.name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {source.name}",
            )

            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            if source.name not in repo_files:
                raise RuntimeError(
                    f"Uploaded file {source.name} is not visible in {repo_id}"
                )

            verified.append(source.name)
            print(f"Verified {source.name} in {repo_id}")
        finally:
            local_path.unlink(missing_ok=True)

    return verified


def main() -> int:
    token = os.environ.get("HF_TOKEN", "")
    repo_id = os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID)
    workdir = Path(os.environ.get("ECUP_MIRROR_DIR", ".ecup-mirror"))

    try:
        verified = mirror_files(repo_id, token, workdir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Mirror complete: {len(verified)} files verified in {repo_id}")
    for name in verified:
        print(f" - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
