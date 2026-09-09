from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.data import discover_layout, load_manifests, resolve_image_paths
from src.transformer_solution import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TRANSFORMER_BACKBONES,
    DualHeadClassifier,
    DualViewTransform,
    blend_head_probabilities,
    combined_classification_loss,
    freeze_backbone,
    unfreeze_matching,
)


class LightingDataset(Dataset):
    def __init__(self, paths, labels=None, image_size=224, train=False, strong=True):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.train = bool(train)
        self.dual = DualViewTransform(image_size=image_size, strong=strong)
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            if self.train:
                clean, augmented = self.dual(image)
                x = (clean, augmented)
            else:
                x = self.eval_transform(image)
        if self.labels is None:
            return x
        return x, int(self.labels[index])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(backbone_key: str) -> tuple[DualHeadClassifier, object]:
    import timm

    spec = TRANSFORMER_BACKBONES[backbone_key]
    backbone = timm.create_model(
        spec.model_name,
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    )
    return DualHeadClassifier(backbone), spec


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_epoch(model, loader, optimizer, device, ordinal_weight, consistency_weight=0.08):
    model.train()
    running = 0.0
    count = 0
    correct = 0
    for (clean, augmented), y in loader:
        clean = clean.to(device)
        augmented = augmented.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        class_clean, ord_clean = model(clean)
        class_aug, ord_aug = model(augmented)
        loss_clean = combined_classification_loss(class_clean, ord_clean, y, ordinal_weight=ordinal_weight)
        loss_aug = combined_classification_loss(class_aug, ord_aug, y, ordinal_weight=ordinal_weight)
        p_clean = blend_head_probabilities(class_clean, ord_clean)
        p_aug = blend_head_probabilities(class_aug, ord_aug)
        consistency = torch.nn.functional.mse_loss(p_clean, p_aug)
        loss = 0.5 * (loss_clean + loss_aug) + consistency_weight * consistency
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running += float(loss.detach()) * len(y)
        pred = (0.5 * (p_clean + p_aug)).argmax(1)
        correct += int((pred == y).sum())
        count += len(y)
    return {"loss": running / max(count, 1), "accuracy": correct / max(count, 1)}


@torch.inference_mode()
def predict(model, loader, device, tta=True):
    model.eval()
    all_probs = []
    all_y = []
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and torch.is_tensor(batch[1]):
            x, y = batch
            all_y.append(y.numpy())
        else:
            x = batch
        x = x.to(device)
        c, o = model(x)
        probs = blend_head_probabilities(c, o)
        if tta:
            xf = torch.flip(x, dims=[3])
            cf, of = model(xf)
            probs = 0.5 * (probs + blend_head_probabilities(cf, of))
        all_probs.append(probs.cpu().numpy())
    probs = np.concatenate(all_probs)
    y = np.concatenate(all_y) if all_y else None
    return probs, y


def cosine_lr(epoch: int, total: int, base: float, floor_ratio: float = 0.08) -> float:
    if total <= 1:
        return base
    phase = epoch / max(total - 1, 1)
    return base * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * phase)))


def fit_one_fold(
    backbone_key: str,
    train_paths,
    y,
    tr_idx,
    va_idx,
    *,
    seed=42,
    head_epochs=1,
    tune_epochs=5,
    batch_size=12,
    workers=2,
    strong=True,
    ordinal_weight=0.35,
    body_lr=2e-5,
    head_lr=3e-4,
):
    set_seed(seed)
    device = torch.device("cpu")
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    model, spec = make_model(backbone_key)
    model.to(device)
    freeze_backbone(model.backbone)

    tr_ds = LightingDataset(
        [train_paths[i] for i in tr_idx], y[tr_idx], spec.image_size, train=True, strong=strong
    )
    va_ds = LightingDataset(
        [train_paths[i] for i in va_idx], y[va_idx], spec.image_size, train=False
    )
    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    va_loader = DataLoader(va_ds, batch_size=max(batch_size, 16), shuffle=False, num_workers=workers)

    history = []
    best_acc = -1.0
    best_state = None
    best_epoch = None

    # Stage A: stabilize the newly initialized heads.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=head_lr, weight_decay=0.02
    )
    for epoch in range(head_epochs):
        stats = run_epoch(model, tr_loader, optimizer, device, ordinal_weight)
        probs, truth = predict(model, va_loader, device, tta=True)
        acc = accuracy_score(truth, probs.argmax(1))
        history.append({"stage": "head", "epoch": epoch + 1, "val_accuracy": float(acc), **stats})
        print(backbone_key, history[-1], flush=True)
        if acc > best_acc:
            best_acc, best_epoch = float(acc), f"head-{epoch + 1}"
            best_state = copy.deepcopy(model.state_dict())

    # Stage B: adapt the final transformer stage while keeping earlier representations stable.
    changed = unfreeze_matching(model.backbone, spec.unfreeze_patterns)
    if changed == 0:
        # Model-name changes in timm must not silently leave the body frozen.
        named = list(model.backbone.named_parameters())
        cutoff = max(0, int(len(named) * 0.85))
        for _, parameter in named[cutoff:]:
            parameter.requires_grad = True
        changed = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)

    body = [p for p in model.backbone.parameters() if p.requires_grad]
    heads = list(model.class_head.parameters()) + list(model.ordinal_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": body, "lr": body_lr},
            {"params": heads, "lr": head_lr},
        ],
        weight_decay=0.03,
    )

    for epoch in range(tune_epochs):
        body_now = cosine_lr(epoch, tune_epochs, body_lr)
        head_now = cosine_lr(epoch, tune_epochs, head_lr)
        optimizer.param_groups[0]["lr"] = body_now
        optimizer.param_groups[1]["lr"] = head_now
        stats = run_epoch(model, tr_loader, optimizer, device, ordinal_weight)
        probs, truth = predict(model, va_loader, device, tta=True)
        acc = accuracy_score(truth, probs.argmax(1))
        row = {
            "stage": "tune",
            "epoch": epoch + 1,
            "val_accuracy": float(acc),
            "body_lr": body_now,
            "head_lr": head_now,
            **stats,
        }
        history.append(row)
        print(backbone_key, row, flush=True)
        if acc > best_acc:
            best_acc, best_epoch = float(acc), f"tune-{epoch + 1}"
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    probs, truth = predict(model, va_loader, device, tta=True)
    return model, spec, probs, truth, {
        "best_accuracy": float(accuracy_score(truth, probs.argmax(1))),
        "best_epoch": best_epoch,
        "history": history,
        "unfrozen_body_parameters": int(changed),
        "trainable_parameters": int(trainable_parameters(model)),
        "confusion_matrix": confusion_matrix(truth, probs.argmax(1)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=sorted(TRANSFORMER_BACKBONES), required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs-transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--safe-aug", action="store_true")
    args = parser.parse_args()

    started = time.time()
    out = Path(args.output) / args.backbone
    out.mkdir(parents=True, exist_ok=True)

    layout = discover_layout(args.data)
    train_df, test_df, _ = load_manifests(layout)
    train_paths, test_paths = resolve_image_paths(layout)
    y = train_df.label.to_numpy(int)

    cv = StratifiedKFold(args.splits, shuffle=True, random_state=args.seed)
    folds = list(cv.split(np.arange(len(y)), y))
    tr_idx, va_idx = folds[args.fold]

    model, spec, val_probs, truth, metrics = fit_one_fold(
        args.backbone,
        train_paths,
        y,
        tr_idx,
        va_idx,
        seed=args.seed,
        head_epochs=args.head_epochs,
        tune_epochs=args.tune_epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        strong=not args.safe_aug,
    )

    device = torch.device("cpu")
    test_ds = LightingDataset(test_paths, labels=None, image_size=spec.image_size, train=False)
    test_loader = DataLoader(test_ds, batch_size=max(args.batch_size, 16), shuffle=False, num_workers=args.workers)
    test_probs, _ = predict(model, test_loader, device, tta=True)

    np.save(out / "val_indices.npy", va_idx)
    np.save(out / "val_probs.npy", val_probs)
    np.save(out / "test_probs.npy", test_probs)
    pd.DataFrame({"id": test_df.id, "label": test_probs.argmax(1)}).to_csv(out / "submission.csv", index=False)
    metrics.update(
        {
            "backbone": args.backbone,
            "model_name": spec.model_name,
            "seed": args.seed,
            "fold": args.fold,
            "splits": args.splits,
            "strong_augmentation": not args.safe_aug,
            "seconds": time.time() - started,
            "validation_rows": int(len(va_idx)),
        }
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
