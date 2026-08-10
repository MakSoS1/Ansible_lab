from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from scripts.lightning_secure_runner import (
    _authenticated_username,
    _teamspace_sort_key,
    decrypt_credentials,
)


_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_TERMINAL_SUCCESS = {"completed", "succeeded", "success"}
_TERMINAL_FAILURE = {"failed", "error", "stopped", "cancelled", "canceled", "deleted"}
_FORBIDDEN_REMOTE_ENV = {"LIGHTNING_API_KEY", "LIGHTNING_USER_ID"}


def probe_command() -> str:
    return "python -c 'import platform; print(\"ecup_lightning_docker_job_ok\", platform.python_version())'"


def _docker_job_kwargs(
    *,
    name: str,
    image: str,
    machine: object,
    command: str,
    teamspace: object,
    username: str,
    env: dict[str, str] | None = None,
    interruptible: bool = True,
) -> dict[str, object]:
    if not _SAFE_NAME_RE.fullmatch(name):
        raise ValueError("job name is not safe")
    if not image or not isinstance(image, str):
        raise ValueError("docker image must be a non-empty string")
    if not username or not _SAFE_NAME_RE.fullmatch(username):
        raise ValueError("username is not safe")
    remote_env = dict(env or {})
    forbidden = _FORBIDDEN_REMOTE_ENV & set(remote_env)
    if forbidden:
        raise ValueError("submitter credential must never be copied into remote job env")
    return {
        "name": name,
        "image": image,
        "machine": machine,
        "command": command,
        "teamspace": teamspace,
        "user": username,
        "env": remote_env,
        "interruptible": bool(interruptible),
    }


def _resolve_teamspace(user: object, username: str):
    teamspaces = sorted(
        list(getattr(user, "teamspaces", [])),
        key=lambda item: _teamspace_sort_key(item, username),
    )
    if not teamspaces:
        raise RuntimeError("authenticated Lightning user has no accessible Teamspace")
    return teamspaces[0]


def _status_text(job: object) -> str:
    raw = getattr(job, "status", "")
    value = getattr(raw, "value", raw)
    return str(value).strip().lower()


def _wait_for_job(job: object, *, timeout_seconds: int, poll_seconds: float = 5.0) -> str:
    deadline = time.monotonic() + timeout_seconds
    last = ""
    while True:
        status = _status_text(job)
        if status and status != last:
            print(f"lightning_job_status={status}", flush=True)
            last = status
        if status in _TERMINAL_SUCCESS:
            return status
        if status in _TERMINAL_FAILURE:
            raise RuntimeError(f"Lightning Docker Job ended with status {status}")
        if time.monotonic() >= deadline:
            try:
                job.stop()
            except Exception:
                pass
            raise TimeoutError(f"Lightning Docker Job exceeded {timeout_seconds}s")
        time.sleep(poll_seconds)


def run_probe(
    *,
    credentials: dict[str, str],
    job_name: str,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Submit a tiny CPU Docker Job without creating or depending on a Studio."""
    os.environ["LIGHTNING_USER_ID"] = credentials["LIGHTNING_USER_ID"]
    os.environ["LIGHTNING_API_KEY"] = credentials["LIGHTNING_API_KEY"]
    job = None
    try:
        from lightning_sdk import Job, Machine, User

        username = _authenticated_username()
        user = User(name=username)
        teamspace = _resolve_teamspace(user, username)
        teamspace_name = str(getattr(teamspace, "name", teamspace))
        print(f"lightning_job_teamspace={teamspace_name}", flush=True)
        kwargs = _docker_job_kwargs(
            name=job_name,
            image="python:3.11-slim",
            machine=Machine.CPU,
            command=probe_command(),
            teamspace=teamspace,
            username=username,
            env={},
            interruptible=True,
        )
        job = Job.run(**kwargs)
        status = _wait_for_job(job, timeout_seconds=timeout_seconds)
        return {
            "mode": "docker-job",
            "job_name": job_name,
            "teamspace": teamspace_name,
            "machine": "CPU",
            "status": status,
            "studio_used": False,
        }
    finally:
        os.environ.pop("LIGHTNING_API_KEY", None)
        os.environ.pop("LIGHTNING_USER_ID", None)
        credentials.clear()
        if job is not None:
            try:
                job.delete()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--ciphertext", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    credentials = decrypt_credentials(args.private_key, args.ciphertext)
    result = run_probe(
        credentials=credentials,
        job_name=args.job_name,
        timeout_seconds=args.timeout_seconds,
    )
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
