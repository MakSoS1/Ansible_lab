import numpy as np
import pytest


def test_ordinal_targets_follow_dark_normal_bright_order():
    torch = pytest.importorskip('torch')
    from src.transformer_solution import ordinal_targets

    y = torch.tensor([0, 1, 2])
    got = ordinal_targets(y)
    expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert torch.equal(got.cpu(), expected)


def test_ordinal_probabilities_are_valid_three_class_distribution():
    torch = pytest.importorskip('torch')
    from src.transformer_solution import ordinal_logits_to_probs

    logits = torch.tensor([[-2.0, -4.0], [2.0, -2.0], [4.0, 2.0]])
    probs = ordinal_logits_to_probs(logits)
    assert probs.shape == (3, 3)
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-6)
    assert probs[0].argmax().item() == 0
    assert probs[1].argmax().item() == 1
    assert probs[2].argmax().item() == 2


def test_combined_loss_backpropagates_through_both_heads():
    torch = pytest.importorskip('torch')
    from src.transformer_solution import combined_classification_loss

    class_logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 2.0]],
        requires_grad=True,
    )
    ordinal_logits = torch.tensor(
        [[-2.0, -4.0], [2.0, -2.0], [4.0, 2.0]],
        requires_grad=True,
    )
    y = torch.tensor([0, 1, 2])
    loss = combined_classification_loss(class_logits, ordinal_logits, y, ordinal_weight=0.35)
    assert torch.isfinite(loss)
    loss.backward()
    assert class_logits.grad is not None
    assert ordinal_logits.grad is not None
    assert torch.isfinite(class_logits.grad).all()
    assert torch.isfinite(ordinal_logits.grad).all()


def test_dual_view_policy_preserves_one_view_and_perturbs_second(tmp_path):
    torch = pytest.importorskip('torch')
    pil = pytest.importorskip('PIL.Image')
    from src.transformer_solution import DualViewTransform

    image = pil.new('RGB', (64, 64), color=(80, 100, 120))
    tfm = DualViewTransform(image_size=64, strong=True, seed=123)
    clean, augmented = tfm(image)
    assert clean.shape == augmented.shape == (3, 64, 64)
    assert torch.isfinite(clean).all() and torch.isfinite(augmented).all()
    assert not torch.allclose(clean, augmented)


def test_backbone_specs_are_transformer_families():
    from src.transformer_solution import TRANSFORMER_BACKBONES

    assert set(TRANSFORMER_BACKBONES) >= {'swin_tiny', 'deit3_small', 'maxvit_tiny'}
    for spec in TRANSFORMER_BACKBONES.values():
        assert spec.model_name
        assert spec.unfreeze_patterns
        assert spec.image_size >= 192
