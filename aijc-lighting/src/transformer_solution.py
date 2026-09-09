from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class TransformerBackboneSpec:
    model_name: str
    image_size: int
    unfreeze_patterns: tuple[str, ...]


TRANSFORMER_BACKBONES: dict[str, TransformerBackboneSpec] = {
    "swin_tiny": TransformerBackboneSpec(
        model_name="swin_tiny_patch4_window7_224.ms_in1k",
        image_size=224,
        unfreeze_patterns=("layers.3", "norm"),
    ),
    "deit3_small": TransformerBackboneSpec(
        model_name="deit3_small_patch16_224.fb_in1k",
        image_size=224,
        unfreeze_patterns=("blocks.10", "blocks.11", "norm"),
    ),
    "maxvit_tiny": TransformerBackboneSpec(
        model_name="maxvit_tiny_tf_224.in1k",
        image_size=224,
        unfreeze_patterns=("stages.3", "norm"),
    ),
}


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    """Encode labels 0/1/2 as the two cumulative events y>0 and y>1."""
    y = y.long()
    return torch.stack(((y > 0).float(), (y > 1).float()), dim=1)


def ordinal_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert cumulative ordinal logits into monotonic three-class probabilities."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("ordinal logits must have shape [N, 2]")
    p_gt0 = torch.sigmoid(logits[:, 0])
    p_gt1 = torch.minimum(torch.sigmoid(logits[:, 1]), p_gt0)
    probs = torch.stack((1.0 - p_gt0, p_gt0 - p_gt1, p_gt1), dim=1)
    probs = probs.clamp_min(1e-8)
    return probs / probs.sum(dim=1, keepdim=True)


def blend_head_probabilities(
    class_logits: torch.Tensor,
    ordinal_logits: torch.Tensor,
    ordinal_mix: float = 0.30,
) -> torch.Tensor:
    """Blend nominal and ordinal heads while retaining a valid probability simplex."""
    mix = float(ordinal_mix)
    if not 0.0 <= mix <= 1.0:
        raise ValueError("ordinal_mix must be within [0, 1]")
    nominal = torch.softmax(class_logits, dim=1)
    ordinal = ordinal_logits_to_probs(ordinal_logits)
    probs = (1.0 - mix) * nominal + mix * ordinal
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def combined_classification_loss(
    class_logits: torch.Tensor,
    ordinal_logits: torch.Tensor,
    y: torch.Tensor,
    ordinal_weight: float = 0.35,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Multiclass CE plus a cumulative ordinal auxiliary objective."""
    ce = F.cross_entropy(class_logits, y.long(), label_smoothing=label_smoothing)
    ord_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets(y))
    return ce + float(ordinal_weight) * ord_loss


class DualHeadClassifier(nn.Module):
    """Wrap a pooled timm-style backbone with nominal and cumulative-ordinal heads."""

    def __init__(self, backbone: nn.Module, dropout: float = 0.15):
        super().__init__()
        if not hasattr(backbone, "num_features"):
            raise ValueError("backbone must expose num_features")
        self.backbone = backbone
        dim = int(backbone.num_features)
        self.dropout = nn.Dropout(float(dropout))
        self.class_head = nn.Linear(dim, 3)
        self.ordinal_head = nn.Linear(dim, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.ndim == 4:
            feat = feat.mean(dim=(-2, -1))
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)
        if feat.ndim != 2:
            feat = feat.flatten(1)
        return feat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.dropout(self.forward_features(x))
        return self.class_head(feat), self.ordinal_head(feat)


def freeze_backbone(backbone: nn.Module) -> None:
    for parameter in backbone.parameters():
        parameter.requires_grad = False


def unfreeze_matching(backbone: nn.Module, patterns: tuple[str, ...]) -> int:
    """Enable gradients only for named parameters matching any requested suffix/stage."""
    changed = 0
    for name, parameter in backbone.named_parameters():
        if any(pattern in name for pattern in patterns):
            if not parameter.requires_grad:
                parameter.requires_grad = True
            changed += parameter.numel()
    return changed


class _RandomGamma:
    def __init__(self, low: float = 0.65, high: float = 1.45, p: float = 0.7):
        self.low = float(low)
        self.high = float(high)
        self.p = float(p)

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        gamma = random.uniform(self.low, self.high)
        return TF.adjust_gamma(image, gamma=gamma, gain=1.0)


class DualViewTransform:
    """Return an exposure-preserving view and a strongly perturbed view."""

    def __init__(self, image_size: int = 224, strong: bool = True, seed: int | None = None):
        self.image_size = int(image_size)
        self.strong = bool(strong)
        self.seed = seed

        self.clean = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        if strong:
            color = transforms.ColorJitter(
                brightness=0.32,
                contrast=0.32,
                saturation=0.15,
                hue=0.025,
            )
            gamma = _RandomGamma(0.65, 1.45, p=0.75)
        else:
            color = transforms.ColorJitter(
                brightness=0.08,
                contrast=0.10,
                saturation=0.05,
                hue=0.01,
            )
            gamma = _RandomGamma(0.88, 1.12, p=0.35)

        self.augmented = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.82, 1.0),
                    ratio=(0.92, 1.08),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(7)], p=0.35),
                color,
                gamma,
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __call__(self, image):
        clean = self.clean(image)
        if self.seed is None:
            return clean, self.augmented(image)

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            augmented = self.augmented(image)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
        return clean, augmented
