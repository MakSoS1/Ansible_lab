from __future__ import annotations

import torch

from ecup_matching.ml.v18_ema import ExponentialMovingAverage


def test_ema_updates_and_copies_trainable_parameters() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 3.0]]))
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 5.0]]))
    ema.update(model)
    state = ema.state_dict()
    assert torch.allclose(state["shadow"]["weight"], torch.tensor([[2.0, 4.0]]))
    with torch.no_grad():
        model.weight.zero_()
    ema.copy_to(model)
    assert torch.allclose(model.weight, torch.tensor([[2.0, 4.0]]))


def test_ema_ignores_frozen_parameters() -> None:
    model = torch.nn.Linear(1, 1)
    model.bias.requires_grad = False
    ema = ExponentialMovingAverage(model, decay=0.9)
    assert "weight" in ema.state_dict()["shadow"]
    assert "bias" not in ema.state_dict()["shadow"]
