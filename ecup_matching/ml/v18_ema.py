from __future__ import annotations

import math


class ExponentialMovingAverage:
    def __init__(self, model, *, decay: float = 0.999):
        import torch

        d = float(decay)
        if not math.isfinite(d) or not 0.0 <= d < 1.0:
            raise ValueError("EMA decay must be finite and in [0,1)")
        self.decay = d
        self.shadow: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                self.shadow[name] = parameter.detach().clone()
        if not self.shadow:
            raise ValueError("EMA requires at least one trainable floating-point parameter")

    def update(self, model) -> None:
        names = set(self.shadow)
        seen: set[str] = set()
        with __import__("torch").no_grad():
            for name, parameter in model.named_parameters():
                if name not in names:
                    continue
                seen.add(name)
                shadow = self.shadow[name]
                shadow.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        if seen != names:
            raise RuntimeError(f"EMA model parameter mismatch: missing={sorted(names-seen)}")

    def copy_to(self, model) -> None:
        names = set(self.shadow)
        seen: set[str] = set()
        with __import__("torch").no_grad():
            for name, parameter in model.named_parameters():
                if name not in names:
                    continue
                seen.add(name)
                parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))
        if seen != names:
            raise RuntimeError(f"EMA model parameter mismatch: missing={sorted(names-seen)}")

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": float(self.decay),
            "shadow": {name: tensor.detach().cpu().clone() for name, tensor in self.shadow.items()},
        }


__all__ = ["ExponentialMovingAverage"]
