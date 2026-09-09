from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
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

        # A local seed makes unit tests and debugging reproducible without permanently
        # changing the caller's random state.
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
