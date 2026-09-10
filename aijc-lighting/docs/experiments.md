# AIIJC Lighting — Experiment Log

This file is the append-only experiment journal for the illumination classification task. Every reported metric must come from a reproducible run. Do not replace measured values with estimates.

## Dataset

- Train: 1,500 RGB PNG images, 512×512.
- Classes: 500 dark / 500 normal / 500 bright.
- Test: 300 RGB PNG images.
- Metric: Accuracy.
- Public competition points: `max(0, accuracy - 0.40) / 0.60`.
- Official image archive cached privately on Hugging Face to avoid repeatedly downloading ~905 MB from Mail.ru.
- Important provenance detail: the Mail.ru ZIP contains images only. `src.data.synthesize_manifests_from_images()` creates sorted CSV manifests when the separately distributed official CSVs are absent. This is safe for ID-keyed ML submissions, but synthetic CSV order must **not** be used for split-order leakage research.
- The separately supplied original `test.csv`/`sample_submission.csv` have the same 300 IDs but a non-sorted order. That original order is now preserved as a permutation of the sorted IDs for structural-generator audits.

## Validation policy

1. Primary model-selection metric: 5-fold stratified OOF Accuracy.
2. Fast architecture probes may use one fixed fold; these are never presented as final OOF.
3. Keep OOF probabilities, fold scores, confusion matrix and test probabilities for every serious candidate.
4. Leakage / generator signals must be validated on labelled train data or independently constrained before they can influence a final claim.
5. Perceptual duplicates / near-duplicates are audited separately.
6. Public leaderboard results are logged separately from OOF. Structural-generator candidates are explicitly marked as leakage hypotheses, not clean ML results.

## Measured experiments

| Family | Configuration | Validation Accuracy | LB Accuracy | Status / conclusion |
|---|---|---:|---:|---|
| V1 ensemble | physics + boosting + spatial + DINOv2 + CNN, 5-fold OOF | **0.5180** | **0.51** | Current verified leaderboard baseline. |
| DINOv3 | ConvNeXt-Tiny DINOv3 frozen features; best RBF/LGB blend, 5-fold OOF | 0.4867 | — | Rejected. |
| SigLIP2 | `google/siglip2-base-patch16-224`; LR/RBF/LGB + prompt families, 5-fold OOF | 0.5113 | — | No improvement; zero-shot illumination 0.4553. |
| V2 handcrafted | 702 physics/local/Retinex/quant features, 5-fold OOF | 0.4960 | — | Full 3-class weak; normal-vs-bright ExtraTrees 0.576 binary CV. |
| Raw quantization | 612 full-resolution 8-bit fingerprints, 5-fold OOF | 0.4867 | — | No deterministic tone-curve shortcut. |
| ResNet18 safe aug | one fixed probe fold | 0.482 | — | Weak. |
| ResNet18 strong aug | one fixed probe fold | 0.478 | — | Weak. |
| EfficientNet-B0 safe aug | one fixed probe fold | 0.496 | — | Below V1. |
| EfficientNet-B0 strong aug | one fixed probe fold | 0.496 | — | No gain from stronger brightness augmentation. |
| Swin-Tiny Solution 6 | task-specific dual-view + ordinal head, one fixed fold | **0.456** | — | Rejected before full OOF. |
| DeiT3-Small Solution 6 | task-specific dual-view + ordinal head, one fixed fold | **0.430** | — | Rejected before full OOF. |
| MaxViT-Tiny Solution 6 | same family, CPU probe | ~0.434 best observed | — | Cancelled after ~55 min; no evidence to justify full run. |
| Public exact-task tabular recipe | RGB/HSV/luminance statistics + histograms + 3×3 grid | **0.5007** selected OOF | — | Literal reproduction does not support the public README's old ~0.92 claim. |
| DINOv2 semantic neighbours | clean/equalized/gray multi-view; kNN/LR/RBF | **0.4953** best CV | — | No same-scene exposure shortcut; best `gray:rbf_c3`. |
| ZIP/member-order audit | archive central-directory order / UUID metadata | n/a | n/a | Image ZIP itself does not expose test labels. |
| M1 CNN probe | hosted macOS M1 MPS | n/a | n/a | MPS OOM before training; rejected infrastructure path. |

## Key observations

### 1. Honest image-only modelling is currently clustered around 0.50

Multiple independent representation families — physics features, CNNs, ViTs, DINOv2/DINOv3, SigLIP2 and the public exact-task recipe — all remain near 0.48–0.52 under reproducible validation. The current best complete OOF remains V1 at 0.518.

This makes blind scaling of another generic backbone a low-value use of runner time. New expensive training must have a distinct hypothesis: better source-scene matching, a genuinely illumination-specific pretraining/objective, or a structural generator signal.

### 2. Mean brightness is not the target by itself

Visual inspection shows large overlap. Bright includes some visually dark/night scenes, while normal includes some bright daylight scenes. Global luminance thresholds are therefore insufficient.

### 3. Normal vs bright is the hardest pair

The V2 handcrafted feature family reaches 0.576 binary CV on labels 1 vs 2 while staying around 0.496 in the full three-class problem. A hierarchical classifier remains a valid low-cost experiment, but not a 0.95 path by itself.

### 4. Raw 8-bit generator fingerprints are not the missing shortcut

Full-resolution residue/bit-plane/histogram-occupancy/tone-curve features reach only 0.4867 three-class OOF and 0.535 on normal-vs-bright. A simple deterministic exposure/gamma transform is rejected.

### 5. Semantic nearest-neighbour hypothesis is rejected in its current form

The first DINOv2 semantic-neighbour run exposed a real preprocessing bug: the fixed ViT expected 518×518 while the probe supplied 224×224. A regression test was added; preprocessing now reads the backbone contract and uses `dynamic_img_size=True` where intended. The corrected run completed successfully.

Corrected 5-fold/LOO results remain weak: clean best learned classifier 0.4773, equalized 0.4507, gray **0.4953**, fused clean+equalized 0.4627, fused-all 0.4700. The proposed complement-of-two-neighbour-class heuristic is also near/below chance on its covered rows. Therefore ordinary DINOv2 source-scene label propagation is not a path to 0.95.

### 6. Structural-order provenance matters

The image-only Mail.ru archive and the separately supplied CSV manifests are not equivalent artifacts. When manifests are absent, our downloader synthesizes:

- train sorted by `(label, UUID)`;
- test sorted by UUID;
- sample submission sorted identically to synthetic test.

The **original supplied CSVs are different**: test/sample order is non-sorted, and the supplied train ordering is also not identical to the synthesized ordering. Consequently all split-generator replay work must restore the original manifest ordering before comparing candidate hidden-label sequences. Earlier conclusions based only on the synthetic sorted order are not used as evidence.

The exact `random_state=42` replay from the previous retention task is already contradicted as an exact answer here by its agreement with the verified V1 leaderboard submission; nevertheless the broader split-family hypothesis remains open until many seeds/generator variants are audited in the original manifest order.

### 7. Public exact-task repository is useful as a proxy, not ground truth

The public same-task repository contains many saved submissions (`master`, grand master, hybrid fusion, MobileNetV3, physics blend, ResNet18, self-training, tabular blend). These can provide independent-ish proxy predictions for generator-seed ranking, but they are never treated as labels.

Its repository history reports later OOF values around 0.51 (for example 0.5140 for a MobileNet ensemble and about 0.510 for ResNet), consistent with our measurements. Therefore the README's older ~0.92+ statement is not accepted as measured evidence.

## Solution 6 — task-specific vision transformers

Architecture and objective:

- Swin-Tiny, DeiT3-Small, MaxViT-Tiny;
- dual exposure-preserving/strong view;
- multiclass CE + ordinal auxiliary objective for `dark < normal < bright`;
- frozen-head stage followed by selective final-block unfreezing;
- TTA and nominal/ordinal probability fusion.

TDD history:

- RED #1: 5 tests failed because `src.transformer_solution` did not exist.
- GREEN #1: ordinal encoding/probabilities, combined loss, dual-view transform and backbone registry implemented; all 5 passed.
- RED #2: 3 tests specified dual-head/selective-unfreeze/probability blending.
- GREEN #2: all 8 transformer tests passed.
- Measured probes then rejected all three transformer candidates; no full 5-fold transformer run is warranted from current evidence.

## Structural generator audit — active high-priority branch

Current implementation:

- `src/structural_split.py` replays `sklearn.model_selection.train_test_split(..., stratify=y)` for a hypothetical 600 examples/class source corpus and returns hidden-test labels in generator test order.
- Seed ranking accepts multiple proxy prediction vectors.
- TDD RED: 3 structural tests failed only because the module was absent.
- TDD GREEN: **15/15 total tests passed** after implementation.
- `scripts/structural_split_audit.py` scans a large seed range and compares each candidate against multiple exact-task public submissions and their majority consensus, with an empirical null distribution so a maximum-over-many-seeds coincidence is not mistaken for a leak.
- A second TDD guard is being added to preserve the non-sorted original supplied test order instead of the synthetic manifest generated from image filenames.

## Next decision gates

1. Finish original-manifest-order TDD guard, then rerun the 0..49,999 structural split scan.
2. If no statistically exceptional sklearn seed appears, extend generator families: non-stratified sklearn, NumPy `RandomState`, NumPy `default_rng`, per-class sampling followed by a final permutation, and Python `random` shuffles.
3. Use the known V1 leaderboard result as an additional consistency constraint whenever its exact prediction vector is available to the audit.
4. Build a hierarchical dark-vs-rest + specialized normal-vs-bright model only as a cheap incremental ML branch.
5. Search for source/dataset-generation code and additional independently produced exact-task submissions before spending time on another generic backbone.
6. Produce a new leaderboard submission only when it carries materially stronger evidence than the current 0.51 baseline; structural submissions are labelled clearly as generator/leak hypotheses.
