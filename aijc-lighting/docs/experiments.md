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
2. Fast architecture probes may use one fixed fold; these are never presented as final OOF.
3. Keep OOF probabilities, fold scores, confusion matrix and test probabilities for every serious candidate.
4. Leakage / generator signals must be validated on labelled train data before they can influence test predictions.
5. Perceptual duplicates / near-duplicates are audited separately.
6. Public leaderboard is used as a final external check, not as a label oracle.

## Measured experiments

| Family | Configuration | Validation Accuracy | LB Accuracy | Status / conclusion |
|---|---|---:|---:|---|
| V1 ensemble | physics + boosting + spatial + DINOv2 + CNN, 5-fold OOF | 0.518 | 0.51 | Baseline ceiling reproduced on leaderboard. |
| DINOv3 | ConvNeXt-Tiny DINOv3 frozen features; best RBF/LGB blend, 5-fold OOF | 0.4867 | — | Worse than V1, rejected. |
| SigLIP2 | `google/siglip2-base-patch16-224`; LR/RBF/LGB + prompt families, 5-fold OOF | 0.5113 | — | No improvement; zero-shot illumination 0.4553. Rejected as frozen-feature solution. |
| V2 handcrafted | 702 physics/local/Retinex/quant features, 5-fold OOF | 0.4960 | — | 3-class weak; specialized normal-vs-bright ExtraTrees reaches 0.576 binary accuracy and may help a hierarchical blend. |
| Raw quantization | 612 full-resolution 8-bit fingerprints, 5-fold OOF | 0.4867 | — | No deterministic tone-curve leak. Best normal-vs-bright binary = 0.535; strongest univariate AUC = 0.5680. |
| ResNet18 safe aug | 1 fixed probe fold | 0.482 | — | Weak. |
| ResNet18 strong aug | 1 fixed probe fold | 0.478 | — | Strong brightness/contrast augmentation does not help this backbone. |
| EfficientNet-B0 safe aug | 1 fixed probe fold | 0.496 | — | Best epoch 7; still below V1. |
| EfficientNet-B0 strong aug | 1 fixed probe fold | 0.496 | — | Best epoch 18; augmentation changed confusion pattern but not accuracy. |
| ZIP-order audit | Central-directory order / class runs / UUID ordering | n/a | n/a | No target-recovery path: test first and sorted by UUID; train stored in three sorted class directories. |
| M1 CNN probe | ResNet18/EfficientNet-B0 on hosted macOS M1 MPS | n/a | n/a | MPS OOM before training; do not use hosted M1 GPU for these runs. |
| Solution 6 transformer | Swin-T / DeiT3-S / MaxViT-T task-specific fine-tuning | running | — | All reusable transformer primitives pass TDD; three backbones are training in parallel. |

## Key observations

### 1. Mean brightness is not the target by itself

Visual inspection shows overlap between the three classes. Bright contains some visually dark/night scenes, while normal contains some bright daylight scenes. Therefore a simple global luminance threshold is insufficient.

### 2. Frozen foundation features alone are not enough so far

DINOv3 and SigLIP2 did not improve the V1 OOF. This motivates task-specific fine-tuning rather than another linear probe.

### 3. Normal vs bright is the hardest pair

The V2 handcrafted probe is substantially more useful when restricted to labels 1 and 2: ExtraTrees reaches 0.576 binary CV while the same feature family reaches only ~0.496 in the full three-class task. Several color-chromaticity and Retinex-distribution features have univariate binary AUC around 0.58. This supports a later hierarchical model, but it is not yet strong enough alone.

### 4. Raw 8-bit generator fingerprints are not the missing shortcut

Full-resolution, pre-resize residue/bit-plane/histogram-occupancy/tone-curve features reach only 0.4867 three-class OOF and 0.535 on normal-vs-bright. The simple deterministic exposure/gamma-generator hypothesis is therefore rejected in its current form.

### 5. Structural leakage audit so far

The official ZIP central directory contains:

- members 0–299: test, lexicographically sorted UUID filenames;
- members 300–799: train/bright;
- members 800–1299: train/dark;
- members 1300–1799: train/normal.

This does not expose test labels. UUID-only and ordinary metadata-only probes are near chance.

### 6. Generic brightness augmentation is not sufficient by itself

On the fixed CNN probe fold, ResNet18 changes from 0.482 (safe augmentation) to 0.478 (strong augmentation), while EfficientNet-B0 is 0.496 under both safe and strong variants. This rejects the idea that simply turning up ColorJitter is enough; representation and objective still matter.

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

TDD status:

- RED #1: 5 tests failed because `src.transformer_solution` did not exist.
- GREEN #1: ordinal encoding/probabilities, combined loss, dual-view transform and backbone registry implemented; all 5 tests passed.
- RED #2: 3 additional tests failed because dual-head wrapper/selective unfreezing/head-probability blend were intentionally absent.
- GREEN #2: all 8 transformer unit tests pass on a clean Ubuntu runner. Training probes are now running in a three-way matrix.

NVIDIA context: DLSS 4 moved from CNNs to vision transformers; this competition branch uses that architectural idea, but the actual model is trained specifically for illumination classification rather than copying a DLSS network.

## Public exact-task reference implementation

A public repository from the same competition contains an implementation for this exact task with EfficientNet/ResNet/MobileNet, 5-fold CV, aggressive augmentation, TTA, handcrafted physics features, ordinal regressors/classifiers, self-training utilities and a multi-model final script. We treat it as an implementation/reference source, not as a source of test labels. Its methods are being reproduced and validated on our own folds before adoption.

## Next decision gates

1. Finish Solution 6 transformer Swin/DeiT3/MaxViT probes in parallel.
2. Promote only candidates clearly above V1 (`>0.518 OOF`, or a convincingly stronger fixed-fold probe) to full 5-fold training.
3. Build a hierarchical dark-vs-rest + specialized normal-vs-bright model if it improves OOF.
4. Search semantic/source-scene neighbours across train/test using strong vision embeddings; label propagation is allowed only if validated on train pseudo-holdouts.
5. Validate balanced class-count assignment on CV folds before applying the likely 100/100/100 test prior.
6. Reproduce the strongest methods from the public exact-task final script under our validation protocol.
7. Search OOF blends and produce a new submission only when measured evidence says it should beat the current 0.51 leaderboard result.
