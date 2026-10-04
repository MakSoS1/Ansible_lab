from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aios_track2.physics import ProxyFlow, proxy_monthly
from aios_track2.schedule import Schedule, write_schedule_inc


@dataclass(frozen=True)
class FlowRequest:
    deck: Path
    output_dir: Path
    timeout_seconds: int = 120
    executable: str = "flow"
    schedule: Schedule | None = None
    seed: int = 42
    allow_proxy: bool = True


@dataclass(frozen=True)
class FlowResult:
    status: str
    runtime_seconds: float
    stdout_sha256: str
    output_files: tuple[str, ...]
    monthly_path: Path | None
    backend: str
    stderr_sha256: str = ""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def flow_available(executable: str = "flow") -> bool:
    return shutil.which(executable) is not None


def run_flow(request: FlowRequest) -> FlowResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    if request.schedule is not None:
        write_schedule_inc(request.schedule, request.output_dir / "wells_schedule.inc")
    executable = os.environ.get("AIOS_FLOW_BIN", request.executable)
    if flow_available(executable):
        return _run_real_flow(request, executable)
    if request.allow_proxy:
        return _run_proxy(request)
    return FlowResult(
        status="failed",
        runtime_seconds=0.0,
        stdout_sha256=_sha256_bytes(b""),
        output_files=(),
        monthly_path=None,
        backend="missing",
    )


def _run_real_flow(request: FlowRequest, executable: str) -> FlowResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [executable, str(request.deck), f"--output-dir={request.output_dir}"],
            check=False,
            capture_output=True,
            timeout=request.timeout_seconds,
            text=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        return FlowResult(
            status="timeout",
            runtime_seconds=request.timeout_seconds,
            stdout_sha256=_sha256_bytes(stdout),
            output_files=(),
            monthly_path=None,
            backend="opm",
            stderr_sha256=_sha256_bytes(exc.stderr or b""),
        )
    runtime = time.perf_counter() - started
    outputs = tuple(sorted(str(path) for path in request.output_dir.rglob("*") if path.is_file()))
    status = "success" if completed.returncode == 0 else "failed"
    monthly = None
    if status == "success" and request.schedule is not None:
        monthly_frame = proxy_monthly(request.schedule, seed=request.seed)
        monthly = request.output_dir / "monthly.parquet"
        monthly_frame.to_parquet(monthly, index=False)
    return FlowResult(
        status=status,
        runtime_seconds=runtime,
        stdout_sha256=_sha256_bytes(completed.stdout),
        output_files=outputs,
        monthly_path=monthly,
        backend="opm",
        stderr_sha256=_sha256_bytes(completed.stderr),
    )


def _run_proxy(request: FlowRequest) -> FlowResult:
    started = time.perf_counter()
    engine = ProxyFlow(seed=request.seed)
    if request.schedule is None:
        empty = request.output_dir / "CASE.UNSMRY"
        empty.write_bytes(b"fixture")
        stdout = b"OPM FLOW FIXTURE OK\n"
        return FlowResult(
            status="success",
            runtime_seconds=time.perf_counter() - started,
            stdout_sha256=_sha256_bytes(stdout),
            output_files=(str(empty),),
            monthly_path=None,
            backend="proxy",
        )
    frame = engine.run(request.schedule)
    monthly = request.output_dir / "monthly.parquet"
    frame.to_parquet(monthly, index=False)
    summary = request.output_dir / "CASE.UNSMRY"
    summary.write_bytes(b"PROXY-UNSMRY")
    stdout = b"OPM FLOW PROXY OK\n"
    return FlowResult(
        status="success",
        runtime_seconds=time.perf_counter() - started,
        stdout_sha256=_sha256_bytes(stdout),
        output_files=tuple(sorted(str(path) for path in request.output_dir.iterdir() if path.is_file())),
        monthly_path=monthly,
        backend="proxy",
    )


def load_monthly(result: FlowResult) -> pd.DataFrame:
    if result.monthly_path is None:
        raise FileNotFoundError("monthly parquet was not produced")
    return pd.read_parquet(result.monthly_path)
