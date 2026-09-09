# AIIJC Lighting Max-Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate and compare five distinct illumination-classification solutions, audit structural leakage, and produce the strongest reproducible competition submission.

**Architecture:** A small Python package under `aijc-lighting/` separates data acquisition, image feature extraction, leakage diagnostics, validation, candidate models and ensemble selection. GitHub Actions downloads the official Mail.ru archive into the checkout, runs tests and experiments, and publishes all metrics/submissions as workflow artifacts without committing the binary dataset to Git history.

**Tech Stack:** Python 3.11, numpy, pandas, Pillow, OpenCV-headless, scikit-image, PyWavelets, scikit-learn, LightGBM, CatBoost, PyTorch, torchvision, timm, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-09-aijc-lighting-max-score-design.md`

## Global Constraints

- Competition classes are exactly `0=dark`, `1=normal`, `2=bright`.
- Primary metric is Accuracy; points are `max(0, accuracy - 0.40) / 0.60`.
- Primary CV is 5-fold stratified with seed 42; group-aware CV overrides it when near-duplicate grouping shows inflation.
- Test labels may not be inferred through leaderboard probing.
- Structural leakage may be used only after it predicts held-out known training labels reproducibly.
- Official data source is `https://cloud.mail.ru/public/GCsv/1BXmZPEBj`.
- Binary image data are downloaded into `aijc-lighting/data/` at runtime and ignored by Git.
- Every candidate must emit OOF predictions, test predictions and a 300-row `id,label` submission.

---

### Task 1: Project skeleton, downloader contract and dataset validation

**Files:**
- Create: `aijc-lighting/requirements.txt`
- Create: `aijc-lighting/.gitignore`
- Create: `aijc-lighting/src/__init__.py`
- Create: `aijc-lighting/tests/test_data.py`
- Create: `aijc-lighting/src/data.py`
- Create: `aijc-lighting/tests/test_download_mailru.py`
- Create: `aijc-lighting/scripts/download_mailru.py`

**Interfaces:**
- Produces: `DatasetLayout`, `discover_layout(root: Path) -> DatasetLayout`, `validate_dataset(layout: DatasetLayout) -> dict`, `normalize_extracted_tree(root: Path) -> DatasetLayout`, `extract_public_id(url: str) -> str`, `build_download_url(public_url: str, dispatcher_payload: dict, token_payload: dict | None) -> str`.

- [ ] **Step 1: Write failing dataset-layout tests**

Create fixtures with synthetic `train.csv`, `test.csv`, class folders and tiny PNG files. Assert `discover_layout` resolves manifests and folders, `validate_dataset` rejects missing IDs, and accepts exactly matched IDs/classes.

- [ ] **Step 2: Run dataset tests and verify RED**

Run: `pytest aijc-lighting/tests/test_data.py -q`
Expected: import failure because `src.data` does not exist.

- [ ] **Step 3: Implement minimal dataset discovery/validation**

Implement immutable `DatasetLayout` paths, CSV schema checks, class-directory mapping and exact manifest/image set equality. Return counts and class distribution from `validate_dataset`.

- [ ] **Step 4: Run dataset tests and verify GREEN**

Run: `pytest aijc-lighting/tests/test_data.py -q`
Expected: all dataset tests pass.

- [ ] **Step 5: Write failing Mail.ru URL tests**

Test public ID extraction for `GCsv/1BXmZPEBj`, dispatcher endpoint selection and deterministic fallback construction. The downloader must reject URLs outside `cloud.mail.ru/public/`.

- [ ] **Step 6: Run downloader tests and verify RED**

Run: `pytest aijc-lighting/tests/test_download_mailru.py -q`
Expected: import failure because `scripts.download_mailru` does not exist.

- [ ] **Step 7: Implement downloader**

Use `urllib.request` only for bootstrap networking: fetch share HTML/dispatcher, fetch `/api/v2/tokens/download` when available, derive `weblink_get`, append public ID, set the share URL as Referer, stream to a temporary ZIP, validate ZIP integrity, extract, normalize tree, then call `validate_dataset`. Keep a fallback direct-download strategy documented in code for current Mail.ru endpoint changes.

- [ ] **Step 8: Run downloader tests and compile check**

Run: `pytest aijc-lighting/tests/test_download_mailru.py aijc-lighting/tests/test_data.py -q && python -m py_compile aijc-lighting/scripts/download_mailru.py aijc-lighting/src/data.py`
Expected: PASS.

- [ ] **Step 9: Commit**

Commit message: `feat(aijc-lighting): add reproducible dataset downloader`

### Task 2: Physics features and leakage audit

**Files:**
- Create: `aijc-lighting/tests/test_features.py`
- Create: `aijc-lighting/src/features.py`
- Create: `aijc-lighting/tests/test_leakage.py`
- Create: `aijc-lighting/src/leakage.py`

**Interfaces:**
- Produces: `extract_global_features(path: Path) -> dict[str, float]`, `extract_spatial_features(path: Path) -> dict[str, float]`, `extract_feature_table(paths: list[Path], mode: str) -> pd.DataFrame`, `phash64(path: Path) -> int`, `build_duplicate_groups(paths: list[Path], max_hamming: int = 4) -> np.ndarray`, `audit_leakage(train_df, test_df, train_paths, test_paths) -> dict`.

- [ ] **Step 1: Write failing global-feature tests**

Create black, mid-gray, white and two-tone images. Assert monotonic luminance means, expected clipping ratios, finite histogram/entropy outputs and deterministic column ordering.

- [ ] **Step 2: Verify RED**

Run: `pytest aijc-lighting/tests/test_features.py -q`
Expected: import failure.

- [ ] **Step 3: Implement global physics features**

Implement RGB luminance, HSV and Lab summaries, percentiles `[1,5,10,25,50,75,90,95,99]`, 32-bin normalized luminance histogram, shadow/highlight fractions, entropy, channel moments, saturation/value interactions, log-luminance and gamma summaries.

- [ ] **Step 4: Implement spatial/wavelet features after adding their failing tests**

Tests assert 4x4 zone count, center-edge contrast behavior on a synthetic center-bright image, finite Sobel/Laplacian values and Haar wavelet energy shape. Implement 4x4/8x8 zone means/stds, gradient statistics, Laplacian variance, Gaussian difference summaries, single-level Haar sub-band energies and simple Retinex residual statistics.

- [ ] **Step 5: Run feature tests**

Run: `pytest aijc-lighting/tests/test_features.py -q`
Expected: PASS.

- [ ] **Step 6: Write failing leakage tests**

Use synthetic UUIDs/labels and exact/near-duplicate images. Assert duplicate grouping joins identical images, Hamming-threshold grouping is deterministic, and audit output includes `uuid`, `metadata`, `duplicates`, `ordering`, `split_reconstruction` sections.

- [ ] **Step 7: Implement leakage audit**

Extract UUID integer chunks/version/variant/prefix features; file size/dimensions/mode/format/metadata keys; cryptographic hash and perceptual hash; train-test nearest-hash distance; row-order/class-run tests; candidate stratified split reconstruction checks for plausible 1800-total balanced data and common seeds `[0,1,7,13,42,99,2024,2025,2026]`. Evaluate any metadata-only classifier only with held-out CV and report its accuracy explicitly.

- [ ] **Step 8: Run leakage tests**

Run: `pytest aijc-lighting/tests/test_leakage.py -q`
Expected: PASS.

- [ ] **Step 9: Commit**

Commit message: `feat(aijc-lighting): add physics features and leakage audit`

### Task 3: Common validation and Solutions 1-3

**Files:**
- Create: `aijc-lighting/tests/test_validation.py`
- Create: `aijc-lighting/src/validation.py`
- Create: `aijc-lighting/tests/test_solutions_tabular.py`
- Create: `aijc-lighting/src/solutions.py`

**Interfaces:**
- Produces: `CVResult(name: str, oof_probs: np.ndarray, test_probs: np.ndarray, fold_scores: list[float], oof_accuracy: float)`, `make_splits(y, groups=None, n_splits=5, seed=42)`, `run_solution_1(...) -> CVResult`, `run_solution_2(...) -> CVResult`, `run_solution_3(...) -> CVResult`.

- [ ] **Step 1: Write failing split tests**

Assert each index appears exactly once in validation, class ratios are approximately preserved, group IDs never cross train/validation when groups are supplied, and seed 42 is reproducible.

- [ ] **Step 2: Verify RED and implement validation**

Run: `pytest aijc-lighting/tests/test_validation.py -q`; then implement StratifiedKFold/StratifiedGroupKFold wrapper and re-run to PASS.

- [ ] **Step 3: Write failing Solution 1 test**

Use a synthetic three-band luminance dataset with clearly separable labels and assert `run_solution_1` produces `(n,3)` OOF probabilities, `(m,3)` test probabilities, rows summing to one and OOF accuracy >0.95.

- [ ] **Step 4: Implement Solution 1**

Standardize global physics features inside each fold. Compare multinomial logistic regression, shrinkage LDA and calibrated linear SVM using inner train-only selection; retain the strongest deterministic configuration across folds.

- [ ] **Step 5: Write failing Solution 2/3 tests**

Assert boosted/spatial models satisfy probability shapes and exceed random accuracy on nonlinear synthetic data.

- [ ] **Step 6: Implement Solution 2**

Train CatBoost, LightGBM and ExtraTrees on global+physics features. Keep fixed conservative hyperparameters and optionally average two models only if the average improves OOF accuracy after all folds are produced.

- [ ] **Step 7: Implement Solution 3**

Train ExtraTrees, HistGradientBoosting and RBF-SVM on spatial/wavelet features; add an ordinal pair of binary classifiers for thresholds `label>0` and `label>1`, enforce monotonic probabilities and include it in OOF comparison.

- [ ] **Step 8: Run tests**

Run: `pytest aijc-lighting/tests/test_validation.py aijc-lighting/tests/test_solutions_tabular.py -q`
Expected: PASS.

- [ ] **Step 9: Commit**

Commit message: `feat(aijc-lighting): add three tabular illumination solutions`

### Task 4: Solutions 4-5, embeddings and CNN

**Files:**
- Modify: `aijc-lighting/src/solutions.py`
- Create: `aijc-lighting/tests/test_vision_helpers.py`

**Interfaces:**
- Produces: `extract_pretrained_embeddings(paths, backbone='efficientnet_b0', batch_size=32, cache_path=None) -> np.ndarray`, `run_solution_4(...) -> CVResult`, `run_solution_5(...) -> CVResult`.

- [ ] **Step 1: Write failing vision-helper tests**

Monkeypatch the backbone constructor with a tiny deterministic torch module, create 8 tiny RGB images, and assert embedding shape/order/cache round-trip without downloading real weights.

- [ ] **Step 2: Verify RED**

Run: `pytest aijc-lighting/tests/test_vision_helpers.py -q`
Expected: missing function failure.

- [ ] **Step 3: Implement cached pretrained embeddings and Solution 4**

Use timm `efficientnet_b0` features with ImageNet normalization and `num_classes=0`; fall back to `mobilenetv3_small_100` if weight download/model creation fails. Fit standardized linear SVM, logistic regression and shallow LightGBM on embedding-only and embedding+physics matrices; select by OOF.

- [ ] **Step 4: Add CNN training helper tests**

Test transform construction to ensure deterministic validation transforms and train transforms that preserve brightness ordering (geometry + small color jitter only). Test model head has exactly 3 outputs.

- [ ] **Step 5: Implement Solution 5**

Fine-tune EfficientNet-B0 with all layers trainable, backbone LR `1e-5`, head LR `3e-4`, AdamW, weight decay `1e-4`, label smoothing `0.05`, cosine schedule, early stopping and 3 TTA transforms. CPU workflow default uses 3 folds and at most 12 epochs; environment variables can raise folds/epochs for stronger hardware. Produce OOF/test probabilities and blend with the strongest non-CNN candidate only after OOF comparison.

- [ ] **Step 6: Run tests and compile check**

Run: `pytest aijc-lighting/tests/test_vision_helpers.py -q && python -m py_compile aijc-lighting/src/solutions.py`
Expected: PASS.

- [ ] **Step 7: Commit**

Commit message: `feat(aijc-lighting): add embedding and CNN candidates`

### Task 5: Ensemble selection, experiment runner and submission integrity

**Files:**
- Create: `aijc-lighting/tests/test_ensemble.py`
- Create: `aijc-lighting/src/ensemble.py`
- Create: `aijc-lighting/run_experiments.py`

**Interfaces:**
- Produces: `select_best(results: list[CVResult], y: np.ndarray, group_scores: dict | None = None) -> dict`, `write_submission(test_ids, probs, path) -> Path`, CLI `python run_experiments.py --data-dir data --output-dir outputs --vision-mode embeddings|full`.

- [ ] **Step 1: Write failing ensemble tests**

Construct synthetic OOF probabilities where one candidate is best and another pair's convex blend improves by one sample. Assert winner selection, deterministic tie-breaking, exactly 300 submission rows, manifest-order IDs and labels restricted to `{0,1,2}`.

- [ ] **Step 2: Verify RED and implement ensemble utilities**

Run: `pytest aijc-lighting/tests/test_ensemble.py -q`; implement bounded alpha grid `[0,0.05,...,1]`, conservative group-score check and submission writer; re-run to PASS.

- [ ] **Step 3: Implement experiment runner**

Runner validates data, resolves image paths from manifests, runs leakage audit, computes/caches feature tables, derives duplicate groups, runs Solutions 1-5, serializes `metrics.json`, `leakage_audit.json`, OOF arrays and six submission files (`solution_1.csv` through `solution_5.csv`, `submission_best.csv`). It prints a sorted score table and the chosen candidate.

- [ ] **Step 4: Run all unit tests**

Run: `pytest aijc-lighting/tests -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat(aijc-lighting): add experiment runner and model selection`

### Task 6: GitHub Actions, documentation and real-data execution

**Files:**
- Create: `.github/workflows/aijc-lighting-max-score.yml`
- Create: `aijc-lighting/README.md`

**Interfaces:**
- Workflow artifacts: `aijc-lighting-results`, containing `outputs/`, normalized manifests and dataset validation report.

- [ ] **Step 1: Add workflow**

Trigger on pushes to `feature/aijc-lighting-max-score` that modify `aijc-lighting/**` or this workflow, plus `workflow_dispatch`. Steps: checkout, Python 3.11, pip cache, install requirements, run pytest, download Mail.ru archive, validate dataset, run experiments with CPU-bounded settings, upload outputs and dataset validation metadata. Upload the source ZIP only when its size is below GitHub artifact limits.

- [ ] **Step 2: Add README**

Document task, exact class mapping, five solutions, leakage policy, local commands, Actions workflow, output files, metric/points formula and how to submit `outputs/submission_best.csv`.

- [ ] **Step 3: Inspect workflow run caused by the commit**

Use GitHub Actions job logs. If downloader fails because Mail.ru changed endpoint shape, patch only downloader URL derivation and re-run. If a package install/model download fails, apply documented fallback without weakening Solutions 1-3.

- [ ] **Step 4: Compare real CV scores**

Read `metrics.json` artifact. Record all five OOF/group-aware accuracies in README and identify the winner. Inspect class distribution, confusion matrix and leakage audit. If a proven leak beats all image models on held-out known labels, add its predictions as an explicitly named structural candidate; otherwise keep image winner.

- [ ] **Step 5: Run verification**

Verify unit tests, workflow status, exact output row count/order/schema, no missing labels and existence of all five candidate submissions.

- [ ] **Step 6: Commit final documentation**

Commit message: `docs(aijc-lighting): record validated experiment results`

### Task 7: Final branch review and PR

**Files:**
- Review all files under `aijc-lighting/`, `.github/workflows/aijc-lighting-max-score.yml`, design and plan documents.

- [ ] **Step 1: Verify branch diff contains no binary dataset**

Use branch comparison and ensure only source, tests, docs and workflow files are committed.

- [ ] **Step 2: Verify completion evidence**

Require passing tests, successful workflow, downloadable artifact and a valid chosen `submission_best.csv`.

- [ ] **Step 3: Open a pull request**

Create PR from `feature/aijc-lighting-max-score` to `main` summarizing five candidates, real validation scores, leakage findings and the winning submission. Do not merge automatically.