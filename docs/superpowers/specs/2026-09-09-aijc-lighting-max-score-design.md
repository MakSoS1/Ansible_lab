# AIIJC Lighting Max-Score Design

## Goal

Build a reproducible competition pipeline for the three-class illumination task (`dark=0`, `normal=1`, `bright=2`) inside `aijc-lighting/`, evaluate five materially different solution families under the same honest validation protocol, audit the dataset for structural leakage, and generate the highest-confidence `submission.csv` for leaderboard upload.

## Repository layout

The project lives under `aijc-lighting/` and must not modify the existing Ansible/AIOS project structure outside a small workflow addition under `.github/workflows/`.

- `aijc-lighting/README.md` — commands, results table, validation notes, leaderboard workflow.
- `aijc-lighting/requirements.txt` — pinned Python dependencies used by CPU GitHub Actions.
- `aijc-lighting/data/` — runtime download/extraction target. Binary images and the source ZIP are ignored by Git.
- `aijc-lighting/scripts/download_mailru.py` — deterministic Mail.ru Cloud downloader for the official public link.
- `aijc-lighting/src/data.py` — manifest loading, image-path resolution and class mapping.
- `aijc-lighting/src/features.py` — global luminance, color, histogram, local-zone, Retinex and wavelet features.
- `aijc-lighting/src/leakage.py` — metadata, UUID/order, near-duplicate and split-structure audit.
- `aijc-lighting/src/validation.py` — one common stratified/group-aware CV interface.
- `aijc-lighting/src/solutions.py` — five candidate families and a common `fit_predict_cv` contract.
- `aijc-lighting/src/ensemble.py` — OOF-based ensemble selection and submission writer.
- `aijc-lighting/run_experiments.py` — end-to-end experiment runner.
- `aijc-lighting/tests/` — unit tests for downloader parsing, feature extraction, validation, leakage checks and submission integrity.
- `.github/workflows/aijc-lighting-max-score.yml` — downloads the official data, runs tests and all five candidates, and uploads metrics/submissions as artifacts.

## Data acquisition

Official source: `https://cloud.mail.ru/public/GCsv/1BXmZPEBj` (`data_освещённость.zip`). The downloader must derive a direct Mail.ru endpoint at runtime instead of hard-coding an expiring storage host. The extracted checkout-local layout is normalized to:

```text
aijc-lighting/data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/
│   ├── dark/
│   ├── normal/
│   └── bright/
└── test/
```

The runtime verifies exactly 1500 labelled images, 300 test images, 500 examples for each training class, manifest/image ID consistency, and no missing files before experiments start.

## Leakage and generator audit

Because the same task authors produced the previous retention dataset with a recoverable split artifact, leakage analysis runs before model selection. It checks:

1. CSV row order and directory enumeration order against class labels.
2. UUID components, prefixes, integer interpretations, version/variant fields and byte-level UUID features against labels.
3. PNG/JPEG dimensions, mode, EXIF/text chunks, encoder metadata, file size, compression signatures and image byte hashes.
4. Exact duplicates and perceptual near-duplicates across train and test.
5. Whether a plausible original `600/600/600` dataset and a stratified 1500/300 split can be reconstructed from ordering or metadata.
6. Whether any structural rule predicts held-out known training labels significantly above chance without inspecting image pixels.

A leakage rule is eligible for the final submission only when it is reproducible and validated on known training labels. Leaderboard probing is not used to infer labels.

## Validation protocol

Primary local metric: five-fold `StratifiedKFold(shuffle=True, random_state=42)` accuracy. If perceptual-hash clustering reveals related frames, a second `StratifiedGroupKFold` score is mandatory and takes precedence for model selection when the ordinary CV score is materially inflated. Each candidate must output OOF class probabilities or decision scores and test class probabilities in manifest order. No train-derived preprocessing is fitted on validation rows.

The results table records mean fold accuracy, OOF accuracy, group-aware accuracy when applicable, training time, and prediction distribution. The winner is selected by the most conservative reliable validation score, not by in-sample accuracy.

## Five solution families

### Solution 1 — global illumination baseline

Extract mean/std luminance, robust percentiles, black/white clipping ratios, RGB/HSV/Lab moments, entropy and histogram bins. Fit regularized multinomial logistic regression, LDA/QDA and calibrated linear SVM; select the strongest OOF member. This establishes how separable the labels are by pure illumination physics.

### Solution 2 — physics/tabular boosting

Extend Solution 1 with log-luminance, channel ratios, saturation/value interactions, local contrast, shadow/highlight fractions, gamma-response summaries and Retinex statistics. Train CatBoost, LightGBM and ExtraTrees with conservative depth/regularization. Blend only when OOF accuracy improves.

### Solution 3 — spatial/zonal/wavelet model

Use 4x4 and 8x8 luminance grids, center/edge/corner statistics, gradient magnitude, Laplacian variance, multi-scale Gaussian differences and Haar wavelet energy per sub-band. Train HistGradientBoosting/ExtraTrees/RBF-SVM and an ordinal regressor that exploits `dark < normal < bright`.

### Solution 4 — pretrained visual embeddings

Use an ImageNet-pretrained lightweight backbone (primary: EfficientNet-B0; fallback: MobileNetV3-Small) only as a frozen feature extractor. Cache pooled embeddings and fit linear SVM, multinomial logistic regression and shallow LightGBM on embeddings plus physics features. This tests semantic/context information without expensive end-to-end training.

### Solution 5 — fine-tuned CNN + TTA ensemble

Fine-tune EfficientNet-B0 (and, when runtime allows, MobileNetV3/ResNet18) with differential learning rates, label smoothing and only geometry-preserving augmentations. Brightness/contrast jitter is intentionally small because illumination is the target. Use horizontal flip/center-crop TTA and blend CNN probabilities with the best physics/embedding model using OOF-selected weights.

## Ensemble policy

All candidate OOF predictions are stored. `src/ensemble.py` evaluates single models and bounded convex blends on OOF predictions; a blend is accepted only if it beats both components by at least one correctly classified OOF sample and does not degrade group-aware accuracy. Final predictions are hard labels because the competition submission schema is `id,label` and metric is accuracy.

## CI/runtime strategy

GitHub Actions runs on Ubuntu CPU. Fast candidates 1-3 always run. Candidate 4 runs pretrained inference in batches with cached weights. Candidate 5 uses a bounded epoch budget and early stopping so the workflow completes on CPU. The workflow uploads `metrics.json`, `oof_predictions.npz`, all five candidate submissions, `submission_best.csv`, leakage audit JSON, and the normalized manifests. The data archive itself is downloaded into the repository checkout path during the workflow and may also be uploaded as an Actions artifact, but it is not committed into Git history.

## Success criteria

- All automated tests pass.
- Official Mail.ru dataset downloads and validates from a clean checkout.
- Five distinct candidates complete and produce valid 300-row submissions.
- Leakage audit explicitly reports every tested structural channel and evidence.
- The final submission is selected from honest OOF/group-aware validation unless a structural leak is proven on known labels.
- The repository contains enough code and documentation to reproduce the chosen submission from scratch.