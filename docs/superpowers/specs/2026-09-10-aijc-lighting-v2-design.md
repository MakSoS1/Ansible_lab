# AIIJC Lighting V2 Max-Score Design

## Goal

Raise the AIIJC lighting classifier materially above the current verified 0.51 leaderboard accuracy, targeting 0.95+ if the data-generating process supports it, while preserving a reproducible validation pipeline and avoiding direct use of published test-label submissions.

## Evidence from V1

The completed V1 run is reproducible and matches leaderboard behavior closely: Solution 1 OOF 0.4947, Solution 2 0.5000, Solution 3 0.4807, DINOv2/physics Solution 4 0.5140, MobileNetV3 fine-tune Solution 5 0.4633, and selected blend 0.5180. The uploaded blend scored 0.51 on the platform. This establishes that the current CV is informative and that the bottleneck is model representation/target structure, not a hidden validation mismatch.

The class-level image statistics show class 0 is globally darker, while classes 1 and 2 overlap strongly in global luminance. Therefore V2 must explicitly decompose the problem instead of forcing one three-way model to solve both an easy dark/non-dark boundary and the harder normal/bright boundary with one representation.

## Architecture

V2 is a parallel experiment system with five independent tracks whose OOF probabilities are persisted and combined only after each track is validated.

### Track A — hierarchical illumination model

Train two binary stages:

1. `dark` versus `{normal, bright}` using global illumination, shadow/highlight clipping, gamma-response and spatial luminance statistics.
2. `normal` versus `bright` using features designed to expose differences beyond mean brightness: channel saturation, color temperature proxies, highlight topology, local contrast, gradient visibility, chromaticity, local dynamic range, Retinex residuals, noise/SNR estimates, FFT/wavelet energy and spatial histograms.

The final three-class probabilities are reconstructed as `P(dark)`, `(1-P(dark))*P(normal|non-dark)`, and `(1-P(dark))*P(bright|non-dark)`.

### Track B — multi-backbone foundation embeddings

Extract and cache representations from at least three materially different pretrained models available through timm/torchvision:

- DINOv2 ViT-S/14;
- ConvNeXt-Tiny ImageNet;
- EfficientNet-B0 ImageNet.

For each backbone evaluate standardized multinomial logistic regression, linear SVM, RBF-SVM on PCA-reduced embeddings, shallow LightGBM and CatBoost. Evaluate embedding-only and embedding+physics matrices. Persist each backbone independently so GitHub Actions can parallelize them.

### Track C — fine-tuned CNNs with target-safe augmentation

Fine-tune EfficientNet-B0, ResNet18 and ConvNeXt-Tiny independently. Do not use brightness or exposure jitter because illumination is the label. Use geometry-only transforms, mild crop, horizontal flip and small rotation. Use progressive unfreezing, differential learning rates, AdamW, cosine scheduling, label smoothing, early stopping, and horizontal-flip TTA. Run each backbone as its own CI job.

### Track D — generator and scene-relationship audit

Expand the structural audit beyond V1:

- inspect ZIP member order and class-directory ordering;
- inspect PNG chunks, textual metadata, ICC/gamma information, encoder fields, file sizes and byte-level signatures;
- search for scene relationships using exposure-normalized image fingerprints, normalized grayscale SSIM-style descriptors, DCT/pHash at multiple thresholds, and cosine-nearest-neighbor structure in foundation embeddings;
- test whether images form triplets or families corresponding to the same underlying scene at different exposure classes;
- test any discovered structural rule only on known training labels before it is allowed to influence test predictions.

Direct reuse of a public ready-made test submission is prohibited. Public code and modeling ideas may be studied and reimplemented.

### Track E — transductive constrained assignment and stacking

Because train is exactly 500/500/500 and test contains 300 images, evaluate the hypothesis that the source corpus is balanced at 600/600/600. The constraint is considered only after simulation on OOF folds: for each validation fold, replace unconstrained argmax with a maximum-score assignment that enforces the known fold class counts, then measure whether accuracy improves. If it improves consistently, apply a 100/100/100 assignment to test probabilities.

Build the final stack from persisted OOF probabilities. Search sparse convex blends and a regularized multinomial meta-model under nested/held-out stacking discipline. Prefer solutions that improve accuracy across multiple random CV seeds rather than a single split.

## Validation

Primary selection metric is repeated stratified OOF accuracy using at least seeds 42, 2026 and 1337 for fast/tabular/embedding models. CNNs may use three folds initially for cost, then the leading CNN is rerun with five folds. Confusion matrices and per-class recall are mandatory, especially the 1-versus-2 subproblem.

The current V1 leaderboard result demonstrates that ordinary five-fold OOF is well aligned with the hidden test, so V2 uses OOF improvement as the main optimization signal. Any technique that only improves train accuracy or relies on test-label probing is rejected.

## Runtime strategy

Use a GitHub Actions matrix rather than one serial job. Ubuntu jobs handle CPU-heavy feature extraction and tabular models; macOS Apple Silicon jobs are used only where PyTorch MPS is stable and memory permits. Caches are keyed by dataset hash/backbone. Each job uploads probabilities rather than only hard-label submissions so a lightweight aggregation job can build ensembles without rerunning feature extraction.

The official 905 MB ZIP remains cached in the private Hugging Face dataset already created by V1, so future CI jobs restore it directly.

## Deliverables

- `aijc-lighting/src/features_v2.py` — advanced illumination and normal-vs-bright features.
- `aijc-lighting/src/hierarchical.py` — two-stage CV model and probability reconstruction.
- `aijc-lighting/src/transductive.py` — constrained class-count assignment and validation helpers.
- `aijc-lighting/scripts/audit_scenes.py` — deeper scene/generator audit.
- `aijc-lighting/run_v2.py` — configurable experiment entry point for a single track/backbone.
- `.github/workflows/aijc-lighting-v2-matrix.yml` — parallel experiment matrix plus aggregation.
- V2 artifacts containing OOF probabilities, test probabilities, metrics and candidate submissions.
- A final validated `submission_v2_best.csv` selected from measured OOF evidence.

## Success criteria

1. All existing tests remain green and new unit tests cover hierarchy probability reconstruction and constrained assignment.
2. Every candidate produces exactly 1500 OOF and 300 test probability rows.
3. At least one V2 candidate must beat the V1 OOF benchmark of 0.518 before it is recommended for leaderboard upload.
4. Continue iterating across the approved tracks while there are evidence-backed changes to test; target 0.95+ accuracy if achievable from the available data and reproducible modeling signal.
