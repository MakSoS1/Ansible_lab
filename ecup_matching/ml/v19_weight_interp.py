from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def interpolate_safetensors(
    base_path: Path,
    target_path: Path,
    output_path: Path,
    *,
    alpha: float,
) -> None:
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must be inside [0, 1]")

    base = load_file(str(Path(base_path)), device="cpu")
    target = load_file(str(Path(target_path)), device="cpu")
    if set(base) != set(target):
        raise ValueError("safetensors keys must match")

    out: dict[str, torch.Tensor] = {}
    for name in sorted(base):
        left = base[name]
        right = target[name]
        if left.shape != right.shape:
            raise ValueError(f"tensor shape mismatch for {name}")
        if left.dtype != right.dtype:
            raise ValueError(f"tensor dtype mismatch for {name}")
        if left.is_floating_point():
            mixed = left.to(torch.float32).mul(1.0 - a).add(right.to(torch.float32), alpha=a)
            out[name] = mixed.to(dtype=left.dtype).contiguous()
        else:
            if not torch.equal(left, right):
                raise ValueError(f"non-floating tensor mismatch for {name}")
            out[name] = left.contiguous()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(output))


__all__ = ["interpolate_safetensors"]
