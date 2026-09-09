# AIIJC 2026 — Lighting Level Classification

Competition pipeline for classifying illumination from an image:

- `0` — dark
- `1` — normal
- `2` — bright
- metric — Accuracy
- points — `max(0, accuracy - 0.40) / 0.60`

The goal of this project is not a single baseline but a controlled comparison of **five materially different solutions** plus a structural leakage audit. Every candidate uses the same manifest order and the same group-aware validation when perceptually similar frames are detected.

## Project structure

```text
aijc-lighting/
├── scripts/download_mailru.py
├── src/
│   ├── data.py
│   ├── ensemble.py
│   ├── features.py
│   ├── leakage.py
│   ├── solutions.py
│   └── validation.py
├── tests/
├── run_experiments.py
├── requirements.txt
└── outputs/                 # generated at runtime
```

## Official dataset

Source: `https://cloud.mail.ru/public/GCsv/1BXmZPEBj`

Download into this checkout:

```bash
cd aijc-lighting
python scripts/download_mailru.py --keep-zip
```

The downloader validates the expected competition shape before any model runs:

- 1500 training images;
- 300 test images;
- exactly 500 training examples per class;
- exact CSV ↔ image ID consistency;
- valid `sample_submission.csv` order.

The binary archive and extracted images are intentionally excluded from Git history. GitHub Actions downloads them into `aijc-lighting/data/` and exposes the source archive/results as workflow artifacts.

## Leakage audit

The same challenge authors previously produced a tabular task with a reversible train/test split artifact, so this project audits structure before spending compute on models. `src/leakage.py` checks:

- UUID fields, UUID bytes and ID prefixes;
- row/class ordering;
- file dimensions, format, image metadata, compressed-file size and byte statistics;
- exact hashes and 64-bit perceptual hashes;
- near-duplicate groups and train↔test nearest perceptual distance;
- compatibility with a balanced 1800-image source dataset and common split seeds.

A structural rule is eligible for use only if it predicts known held-out training labels reproducibly. Leaderboard probing is not used to manufacture labels.

## Five solutions

### 1. Global illumination physics

Global luminance/RGB/HSV/Lab moments, robust percentiles, black/white clipping, entropy, histograms, channel ratios, log-luminance and gamma response. Models: regularized logistic regression, shrinkage LDA and linear SVM.

### 2. Physics + tabular boosting

Global features plus spatial/Retinex/wavelet statistics. Models: CatBoost, LightGBM and ExtraTrees with OOF-selected blending.

### 3. Spatial / zonal / wavelet model

4×4 and 8×8 illumination maps, center-vs-edge brightness, Sobel/Laplacian statistics, Gaussian residuals, Haar energies and Retinex. Models: ExtraTrees, HistGradientBoosting, RBF-SVM and an ordinal `dark < normal < bright` classifier.

### 4. Pretrained visual embeddings

Frozen ImageNet EfficientNet-B0 embeddings, with a torchvision fallback, classified by linear SVM, logistic regression and shallow LightGBM. A second branch concatenates embeddings with illumination physics.

### 5. Fine-tuned CNN + TTA

ImageNet EfficientNet-B0, differential learning rates, AdamW, label smoothing, cosine decay, early stopping, weak brightness jitter and horizontal-flip TTA. Brightness augmentation is deliberately small because illumination itself is the target.

## Validation

Primary protocol:

```text
5-fold stratified CV, seed=42
```

Before splitting, pHash-near frames are grouped. If non-singleton groups exist, `StratifiedGroupKFold` is used so variants of one scene cannot leak across train/validation. All final ensemble weights are chosen only from OOF predictions.

## Run locally

```bash
cd aijc-lighting
python -m pip install -r requirements.txt
pytest tests -q
python scripts/download_mailru.py --keep-zip
python run_experiments.py --data-dir data --output-dir outputs --vision-mode full
```

For a quick CPU debugging pass:

```bash
AIJC_CNN_EPOCHS=3 AIJC_CNN_FOLDS=2 \
python run_experiments.py --data-dir data --output-dir outputs --vision-mode full --fast-tabular
```

## Outputs

A full run creates:

```text
outputs/
├── dataset_validation.json
├── leakage_audit.json
├── metrics.json
├── oof_predictions.npz
├── solution_1.csv
├── solution_2.csv
├── solution_3.csv
├── solution_4.csv
├── solution_5.csv
└── submission_best.csv
```

`submission_best.csv` is the leaderboard file. The runner selects it from OOF accuracy and also searches conservative probability blends.

## GitHub Actions

Workflow: `.github/workflows/aijc-lighting-max-score.yml`

It runs from a clean checkout, downloads the official Mail.ru dataset, executes the unit tests and five candidates, and uploads:

- `aijc-lighting-results` — metrics, audit, OOF predictions and all submissions;
- `aijc-lighting-dataset-source` — official ZIP plus validation metadata.

This makes the competition result reproducible without adding hundreds of binary images to the repository history.
