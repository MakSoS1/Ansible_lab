# AIIJC Lighting — Experiment Log

This file is the append-only experiment journal for the illumination classification task. Every reported metric must come from a reproducible run. Do not replace measured values with estimates.

## Dataset

- Train: 1,500 RGB PNG images, 512×512.
- Classes: 500 dark / 500 normal / 500 bright.
- Test: 300 RGB PNG images.
- Metric: Accuracy.
- Public competition points: `max(0, accuracy - 0.40) / 0.60`.
- Official archive cached privately on Hugging Face to avoid repeatedly downloading ~905 MB from Mail.ru.

## Validation policy

1. Primary model-selection metric: 5-fold stratified OOF Accuracy.
2. Keep OOF probabilities, fold scores, confusion matrix and test probabilities for every serious candidate.
3. Leakage / generator signals must be validated on labelled train data before they can influence test predictions.
4. Perceptual duplicates / near-duplicates are audited separately.
5. Public leaderboard is used as a final external check, not as a label oracle.

## Measured experiments

| Family | Configuration | OOF Accuracy | LB Accuracy | Status / conclusion |
|---|---|---:|---:|---|
| V1 ensemble | physics + boosting + spatial + DINOv2 + CNN | 0.518 | 0.51 | Baseline ceiling reproduced on leaderboard. |
| DINOv3 | ConvNeXt-Tiny DINOv3 frozen features; best RBF/LGB blend | 0.4867 | — | Worse than V1, rejected. |
| SigLIP2 | `google/siglip2-base-patch16-224`; LR/RBF/LGB + prompt families | 0.5113 | — | No improvement; zero-shot illumination 0.4553. Rejected as frozen-feature solution. |
| ZIP-order audit | Central-directory order / class runs / UUID ordering | n/a | n/a | No target-recovery path: test first and sorted by UUID; train stored in three sorted class directories. |
| M1 CNN probe | ResNet18/EfficientNet-B0 on hosted macOS M1 MPS | n/a | n/a | MPS OOM before training; do not use hosted M1 GPU for these runs. |
| Raw quantization | Full-resolution 8-bit fingerprints | running | — | Tests deterministic tone/gamma transform artifacts without resize. |
| V2 feature probe | Physics + local illumination + quantization features | running | — | Focuses on normal-vs-bright confusion. |
| CNN augmentation | ResNet18/EfficientNet-B0, safe vs strong brightness/contrast | running | — | Tests organizer-recommended brightness/contrast augmentation. |

## Key observations

### 1. Mean brightness is not the target by itself

Visual inspection shows overlap between the three classes. Bright contains some visually dark/night scenes, while normal contains some bright daylight scenes. Therefore a simple global luminance threshold is insufficient.

### 2. Frozen foundation features alone are not enough so far

DINOv3 and SigLIP2 did not improve the V1 OOF. This motivates task-specific fine-tuning rather than another linear probe.

### 3. Structural leakage audit so far

The official ZIP central directory contains:

- members 0–299: test, lexicographically sorted UUID filenames;
- members 300–799: train/bright;
- members 800–1299: train/dark;
- members 1300–1799: train/normal.

This does not expose test labels. UUID-only and ordinary metadata-only probes are near chance.

## Solution 6 — task-specific vision transformers

Approved design:

- Swin-Tiny as the main local-attention transformer.
- DeiT3-Small as a global ViT-style alternative.
- MaxViT-Tiny as a hybrid local/global attention alternative.
- Dual-view training: one exposure-preserving view + one strong brightness/contrast/gamma view.
- Multiclass cross entropy plus an ordinal auxiliary objective for the ordered labels `dark < normal < bright`.
- Stage A: train classification/ordinal heads with the backbone frozen.
- Stage B: unfreeze the final transformer stage/blocks with a lower learning rate.
- Quick 1-fold probe for all backbones; only winners receive full OOF training.
- TTA at inference.
- Final probability blending is accepted only if it improves OOF.

NVIDIA context: DLSS 4 moved from CNNs to vision transformers; this competition branch uses that architectural idea, but the actual model is trained specifically for illumination classification rather than copying a DLSS network.

## Next decision gates

1. Finish raw-quant, V2 features and CNN augmentation probes.
2. Run Solution 6 transformer probes.
3. Promote only candidates clearly above V1 (`>0.518 OOF`) to full 5-fold training.
4. Search OOF blends and, when justified, balanced 100/100/100 test assignment.
5. Produce a new submission only when measured evidence says it should beat the current 0.51 leaderboard result.
