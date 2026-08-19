import types

import pytest

torch = pytest.importorskip("torch")

from ecup_matching.ml.v20_neural import V20MultiTaskModel, compute_v20_loss, production_base_model


class DummyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=4)
        self.encoder = torch.nn.Linear(4, 4)
        self.classifier = torch.nn.Linear(4, 1)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, return_dict=True, **kwargs):
        x = torch.nn.functional.one_hot(input_ids[:, 0] % 4, num_classes=4).float()
        h = self.encoder(x)
        return types.SimpleNamespace(logits=self.classifier(h), hidden_states=(h.unsqueeze(1),))


def test_missing_auxiliary_labels_do_not_change_data_only_loss():
    outputs = {
        "match_logits": torch.tensor([0.2, -0.3], requires_grad=True),
        "model_conflict_logits": torch.tensor([1.0, 1.0], requires_grad=True),
        "numeric_conflict_logits": torch.tensor([1.0, 1.0], requires_grad=True),
        "variant_conflict_logits": torch.tensor([1.0, 1.0], requires_grad=True),
        "accessory_logits": torch.tensor([1.0, 1.0], requires_grad=True),
        "reason_logits": torch.zeros((2, 11), requires_grad=True),
    }
    target = torch.tensor([1.0, 0.0])
    weights = torch.ones(2)
    zero_mask = torch.zeros(2)
    loss = compute_v20_loss(
        outputs, target, weights,
        aux_targets={}, aux_mask=zero_mask,
        lambda_reason=0.15, lambda_consistency=0.05,
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(outputs["match_logits"], target)
    assert torch.allclose(loss["total"], expected)


def test_source_weights_affect_match_loss():
    outputs = {"match_logits": torch.tensor([-4.0, -4.0])}
    target = torch.tensor([1.0, 0.0])
    a = compute_v20_loss(outputs, target, torch.tensor([1.0, 0.1]), aux_targets={}, aux_mask=torch.zeros(2))
    b = compute_v20_loss(outputs, target, torch.tensor([0.1, 1.0]), aux_targets={}, aux_mask=torch.zeros(2))
    assert float(a["match"]) > float(b["match"])


def test_multitask_model_exposes_aux_heads_but_production_is_base_model():
    base = DummyBase()
    model = V20MultiTaskModel(base, reason_classes=11)
    result = model(input_ids=torch.tensor([[1], [2]]), attention_mask=torch.ones((2, 1)))
    assert result["match_logits"].shape == (2,)
    assert result["reason_logits"].shape == (2, 11)
    assert production_base_model(model) is base
