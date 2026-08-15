from types import SimpleNamespace

import torch

from ecup_matching.v15_model import V15Matcher


class TinyBackbone(torch.nn.Module):
    def __init__(self, hidden_size=12):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.emb = torch.nn.Embedding(64, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.proj(self.emb(input_ids))
        return SimpleNamespace(last_hidden_state=x)


def _inputs(batch=4, seq=7):
    return {
        "input_ids": torch.randint(0, 64, (batch, seq)),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
    }


def test_v15_model_outputs_one_logit_and_owns_one_backbone():
    model = V15Matcher(TinyBackbone(), typed_feature_dim=0, num_categories=20)
    logits = model(**_inputs(), typed_features=None, category_ids=None)
    assert logits.shape == (4,)
    backbone_like = [name for name, _ in model.named_modules() if name == "backbone"]
    assert backbone_like == ["backbone"]


def test_v15_model_typed_feature_fusion_and_category_residual_are_optional():
    model = V15Matcher(
        TinyBackbone(),
        typed_feature_dim=13,
        num_categories=20,
        use_typed_features=True,
        use_category_head=True,
    )
    typed = torch.randn(4, 13)
    category_ids = torch.tensor([0, 1, 1, 19])
    logits = model(**_inputs(), typed_features=typed, category_ids=category_ids)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_v15_model_rejects_missing_inputs_when_optional_heads_enabled():
    model = V15Matcher(TinyBackbone(), typed_feature_dim=13, num_categories=20, use_typed_features=True)
    try:
        model(**_inputs(), typed_features=None, category_ids=None)
    except ValueError as exc:
        assert "typed_features" in str(exc)
    else:
        raise AssertionError("expected missing typed_features to fail closed")
