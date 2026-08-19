from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ALLOWED_BRANCH = "ecup-matching-2026"
DISPATCH_REPOSITORY = "MakSoS1/gpu-dispatch"
SOURCE_SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
PROFILES = ("gpu-check", "smoke", "train")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def resolve_pushed_head(repo: Path) -> str:
    branch = _git(repo, "branch", "--show-current")
    if branch != ALLOWED_BRANCH:
        raise RuntimeError(f"switch to {ALLOWED_BRANCH} before dispatching GPU work")

    source_sha = _git(repo, "rev-parse", "HEAD").lower()
    if SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise RuntimeError("Git did not return an exact 40-character commit SHA")

    remote_line = _git(repo, "ls-remote", "--exit-code", "origin", f"refs/heads/{ALLOWED_BRANCH}")
    remote_sha = remote_line.split(maxsplit=1)[0].lower() if remote_line else ""
    if remote_sha != source_sha:
        raise RuntimeError(
            f"push {ALLOWED_BRANCH} first; local HEAD is not the current remote branch tip"
        )
    return source_sha


def build_workflow_command(profile: str, source_sha: str) -> list[str]:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError("source_sha must be an exact 40-character hexadecimal commit SHA")
    return [
        "gh",
        "workflow",
        "run",
        "ecup-gpu.yml",
        "--repo",
        DISPATCH_REPOSITORY,
        "-f",
        f"source_sha={source_sha.lower()}",
        "-f",
        f"profile={profile}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch a fixed isolated E-CUP GPU profile")
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    source_sha = resolve_pushed_head(args.repo.resolve())
    subprocess.run(build_workflow_command(args.profile, source_sha), check=True)
    print(f"queued {args.profile} for {source_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

