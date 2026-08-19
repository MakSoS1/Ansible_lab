from __future__ import annotations

import hashlib
import math


def _uniform01(*, seed: int, epoch: int, index: int, stream: str) -> float:
    payload = f"{int(seed)}|{int(epoch)}|{int(index)}|{stream}".encode("utf-8")
    raw = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(raw, "big") / float(1 << 64)


def deterministic_decision(
    *,
    seed: int,
    epoch: int,
    index: int,
    stream: str,
    probability: float,
) -> bool:
    p = float(probability)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("probability must be finite and in [0,1]")
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return _uniform01(seed=seed, epoch=epoch, index=index, stream=stream) < p


def augment_serialized_view(
    text: str,
    *,
    drop_residual: bool,
    drop_numeric: bool,
) -> str:
    lines = str(text).splitlines()
    has_identity_anchor = any(
        line.startswith("[MODEL]") or line.startswith("[IDENTITY]") for line in lines
    )
    out: list[str] = []
    for line in lines:
        if drop_residual and line.startswith("[RESIDUAL]"):
            continue
        if drop_numeric and has_identity_anchor and line.startswith("[NUMERIC]"):
            continue
        out.append(line)
    return "\n".join(out)


__all__ = ["augment_serialized_view", "deterministic_decision"]
