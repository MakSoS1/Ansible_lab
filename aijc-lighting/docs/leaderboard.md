# AIIJC Lighting — Scoreboard

Only measured results belong here.

## Competition leaderboard submissions

| Timestamp | Submission | Local OOF | Leaderboard Accuracy | Points | Notes |
|---|---|---:|---:|---:|---|
| 2026-09-10 01:40 | V1 `submission_best.csv` | 0.518 | 0.51 | 0.183 | User-confirmed competition result. Current external baseline. |

## Internal 5-fold CV leaderboard

| Rank | Experiment | OOF Accuracy | Decision |
|---:|---|---:|---|
| 1 | V1 physics/DINOv2/CNN ensemble | 0.5180 | baseline |
| 2 | SigLIP2 learned/prompt blend | 0.5113 | reject standalone |
| 3 | V2 handcrafted physics/local/Retinex/quant | 0.4960 | use only specialized 1-vs-2 component if useful |
| 4 | DINOv3 ConvNeXt-Tiny blend | 0.4867 | reject |
| 4 | Raw full-resolution quantization blend | 0.4867 | reject as standalone generator solution |

## Fixed-fold architecture probes

These values are useful for triage but are not directly comparable to full OOF.

| Experiment | Probe Accuracy | Decision |
|---|---:|---|
| EfficientNet-B0 safe augmentation | 0.496 | weak |
| EfficientNet-B0 strong augmentation | 0.496 | weak |
| ResNet18 safe augmentation | 0.482 | reject |
| ResNet18 strong augmentation | 0.478 | reject |

Transformer Swin-T / DeiT3-S / MaxViT-T probes are currently running and will be added only after successful completion.

## Target thresholds

- Points > 0.30 requires Accuracy > 0.58.
- "Good" in task statement: Accuracy ≥ 0.70.
- "Excellent" in task statement: Accuracy ≥ 0.90.
- Project stretch target requested by the user: Accuracy ≥ 0.95.

The 0.95 target is a goal, not a claimed result. Every improvement must be measured before being promoted to a submission.
