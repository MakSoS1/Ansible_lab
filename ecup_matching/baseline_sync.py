from __future__ import annotations

import os
import sys
from pathlib import Path

from ecup_matching.hf_sync import DEFAULT_REPO_ID, SourceFile, mirror_files


BASELINE_SOURCES: tuple[SourceFile, ...] = (
    SourceFile(
        "baselines/matching-baseline-submit.zip",
        "https://storage.yandexcloud.net/ozon-ecup-2026/Matching/matching-baseline-submit.zip",
    ),
    SourceFile(
        "baselines/matching-baseline-lightweight.zip",
        "https://storage.yandexcloud.net/ozon-ecup-2026/Matching/matching-baseline-lightweight.zip",
    ),
)


def main() -> int:
    token = os.environ.get("HF_TOKEN", "")
    repo_id = os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID)
    workdir = Path(os.environ.get("ECUP_BASELINE_MIRROR_DIR", ".ecup-baseline-mirror"))

    try:
        verified = mirror_files(
            repo_id=repo_id,
            token=token,
            workdir=workdir,
            sources=BASELINE_SOURCES,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Baseline mirror complete: {len(verified)} files verified in {repo_id}")
    for name in verified:
        print(f" - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
