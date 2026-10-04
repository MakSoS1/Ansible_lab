from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True, slots=True)
class ComputeBenchmark:
    platform: str
    machine: str
    torch_version: str
    device: str
    matmul_ms: float
    sequence_ms: float


def run_compute_benchmark(seed: int = 42) -> ComputeBenchmark:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    _ = a @ b
    sync()
    t0 = time.perf_counter()
    for _ in range(5):
        _ = a @ b
    sync()
    matmul_ms = (time.perf_counter() - t0) * 1000 / 5

    gru = torch.nn.GRU(32, 64, batch_first=True).to(device)
    x = torch.randn(32, 48, 32, device=device)
    _ = gru(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = gru(x)
    sync()
    sequence_ms = (time.perf_counter() - t0) * 1000 / 10
    return ComputeBenchmark(platform.system(), platform.machine(), torch.__version__, str(device), matmul_ms, sequence_ms)


def benchmark_json() -> str:
    return json.dumps(asdict(run_compute_benchmark()), indent=2, sort_keys=True)
