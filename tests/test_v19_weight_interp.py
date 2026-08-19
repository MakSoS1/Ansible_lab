from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ecup_matching.ml.v19_weight_interp import interpolate_safetensors


def test_interpolate_safetensors_linear_and_exact_endpoints() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        base = root / "base.safetensors"
        target = root / "target.safetensors"
        mid = root / "mid.safetensors"
        zero = root / "zero.safetensors"
        one = root / "one.safetensors"
        save_file({"w": torch.tensor([0.0, 2.0]), "b": torch.tensor([4.0])}, str(base))
        save_file({"w": torch.tensor([2.0, 6.0]), "b": torch.tensor([8.0])}, str(target))

        interpolate_safetensors(base, target, mid, alpha=0.25)
        interpolate_safetensors(base, target, zero, alpha=0.0)
        interpolate_safetensors(base, target, one, alpha=1.0)

        mid_state = load_file(str(mid))
        assert torch.equal(mid_state["w"], torch.tensor([0.5, 3.0]))
        assert torch.equal(mid_state["b"], torch.tensor([5.0]))
        assert torch.equal(load_file(str(zero))["w"], load_file(str(base))["w"])
        assert torch.equal(load_file(str(one))["w"], load_file(str(target))["w"])


def test_interpolate_safetensors_rejects_mismatched_keys_and_alpha() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        base = root / "base.safetensors"
        target = root / "target.safetensors"
        out = root / "out.safetensors"
        save_file({"w": torch.tensor([1.0])}, str(base))
        save_file({"x": torch.tensor([1.0])}, str(target))
        try:
            interpolate_safetensors(base, target, out, alpha=0.5)
        except ValueError as exc:
            assert "keys" in str(exc)
        else:
            raise AssertionError("mismatched keys must fail")

        save_file({"w": torch.tensor([2.0])}, str(target))
        for alpha in (-0.01, 1.01):
            try:
                interpolate_safetensors(base, target, out, alpha=alpha)
            except ValueError as exc:
                assert "alpha" in str(exc)
            else:
                raise AssertionError("out-of-range alpha must fail")
