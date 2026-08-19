from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from ecup_matching.ml.v18_ema import ExponentialMovingAverage
from ecup_matching.ml.v18_neural import train_pair_phase_v18


class DummyTokenizer:
    def __call__(self, left, right, *, padding, truncation, max_length, return_tensors):
        assert padding and truncation and return_tensors == "pt"
        rows = []
        for a, b in zip(left, right):
            rows.append([min(len(a), 63), min(len(b), 63), (len(a) + len(b)) % 63 + 1])
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}


class TinyPairModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)
        self.classifier = torch.nn.Linear(8, 1)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids).mean(dim=1)
        return SimpleNamespace(logits=self.classifier(hidden))


def test_tiny_v18_train_step_runs_on_mps_when_available_else_cpu() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TinyPairModel().to(device)
    ema = ExponentialMovingAverage(model, decay=0.9)
    frame = pd.DataFrame(
        {
            "id1": [1, 3, 5, 7, 1, 3, 5, 7],
            "id2": [2, 4, 6, 8, 4, 6, 8, 2],
            "target": [0.95, 0.05, 0.90, 0.10, 0.08, 0.92, 0.12, 0.88],
            "weak_weight": [1.0, 1.0, 0.7, 0.7, 0.8, 0.8, 0.6, 0.6],
            "category": ["a", "a", "b", "b", "a", "a", "b", "b"],
        }
    )
    texts = {
        i: "\n".join(
            [
                "[CAT] test",
                f"[NAME] item {i}",
                f"[MODEL] m{i}",
                f"[IDENTITY] count=count_{i}",
                f"[NUMERIC] count_{i}",
                f"[RESIDUAL] token={i}",
            ]
        )
        for i in range(1, 9)
    }
    result = train_pair_phase_v18(
        model=model,
        tokenizer=DummyTokenizer(),
        frame=frame,
        texts=texts,
        device=device,
        phase="v18-tiny-smoke",
        epochs=0.5,
        physical_batch_size=2,
        effective_batch_size=2,
        max_length=16,
        learning_rate=1e-3,
        ranking_weight=0.1,
        seed=2026,
        weak=True,
        pair_swap_probability=0.5,
        residual_dropout_probability=0.15,
        numeric_dropout_probability=0.05,
        ema=ema,
        telemetry_every_steps=0,
    )
    assert result.optimizer_steps > 0
    assert result.examples_seen > 0
    assert ema.state_dict()["shadow"]
