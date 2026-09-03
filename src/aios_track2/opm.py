from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FlowRequest:
    deck: Path
    output_dir: Path
    executable: tuple[str, ...] = ("flow",)
    extra_args: tuple[str, ...] = ()
    timeout_seconds: int = 7200


@dataclass(frozen=True, slots=True)
class FlowResult:
    status: str
    returncode: int | None
    runtime_seconds: float
    stdout_sha256: str
    stderr_sha256: str
    output_files: tuple[Path, ...]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_flow(request: FlowRequest) -> FlowResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    exe_name = Path(request.executable[0]).name.lower()
    if exe_name in {"flow", "flow_gaswater_dissolution_diffusion"}:
        cmd = [*request.executable, str(request.deck), f"--output-dir={request.output_dir}", *request.extra_args]
    else:
        cmd = [*request.executable, str(request.deck), str(request.output_dir), *request.extra_args]
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, shell=False, check=False, capture_output=True, timeout=request.timeout_seconds)
        status = "success" if proc.returncode == 0 else "failed"
        rc: int | None = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        status, rc = "timeout", None
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
    runtime = time.monotonic() - started
    (request.output_dir / "flow.stdout.log").write_bytes(stdout)
    (request.output_dir / "flow.stderr.log").write_bytes(stderr)
    files = tuple(sorted((p for p in request.output_dir.rglob("*") if p.is_file()), key=lambda p: str(p)))
    return FlowResult(status, rc, runtime, _sha(stdout), _sha(stderr), files)
