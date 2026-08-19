from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.gpu_dispatch import (
    ALLOWED_BRANCH,
    DISPATCH_REPOSITORY,
    build_workflow_command,
    resolve_pushed_head,
)


SHA = "a" * 40


def completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def test_build_workflow_command_uses_only_fixed_profile_and_exact_sha() -> None:
    command = build_workflow_command("smoke", SHA)

    assert command == [
        "gh",
        "workflow",
        "run",
        "ecup-gpu.yml",
        "--repo",
        DISPATCH_REPOSITORY,
        "-f",
        f"source_sha={SHA}",
        "-f",
        "profile=smoke",
    ]


@pytest.mark.parametrize("profile", ["gpu-check", "smoke", "train"])
def test_build_workflow_command_accepts_every_fixed_profile(profile: str) -> None:
    assert build_workflow_command(profile, SHA)[-1] == f"profile={profile}"


@pytest.mark.parametrize("source_sha", ["main", "a" * 39, "g" * 40, "a" * 41])
def test_build_workflow_command_rejects_non_commit_input(source_sha: str) -> None:
    with pytest.raises(ValueError, match="40-character"):
        build_workflow_command("smoke", source_sha)


def test_resolve_pushed_head_requires_allowed_branch_and_matching_remote() -> None:
    with patch("scripts.gpu_dispatch.subprocess.run") as run:
        run.side_effect = [
            completed(f"{ALLOWED_BRANCH}\n"),
            completed(f"{SHA}\n"),
            completed(f"{SHA}\trefs/heads/{ALLOWED_BRANCH}\n"),
        ]

        assert resolve_pushed_head(Path("/repo")) == SHA

    assert all(call.kwargs.get("shell") is not True for call in run.call_args_list)


def test_resolve_pushed_head_rejects_unpushed_commit() -> None:
    with patch("scripts.gpu_dispatch.subprocess.run") as run:
        run.side_effect = [
            completed(f"{ALLOWED_BRANCH}\n"),
            completed(f"{SHA}\n"),
            completed(f"{'b' * 40}\trefs/heads/{ALLOWED_BRANCH}\n"),
        ]

        with pytest.raises(RuntimeError, match="push"):
            resolve_pushed_head(Path("/repo"))

