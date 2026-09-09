# AIIJC Lighting V2 Max-Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the V1 0.518 OOF ceiling with a parallel, hierarchy-aware and multi-backbone pipeline, validate constrained assignment, deepen generator/scene auditing, and produce a stronger leaderboard-ready submission.

**Architecture:** V2 decomposes `dark` from the harder `normal vs bright` distinction, evaluates several frozen/fine-tuned visual backbones in parallel, persists OOF/test probabilities, and aggregates them with OOF-selected blends. Structural/transductive techniques are only enabled when they improve known-label validation.

**Tech Stack:** Python 3.11, NumPy, pandas, scikit-learn, LightGBM, CatBoost, PyTorch/torchvision, timm, Pillow, OpenCV, GitHub Actions, Hugging Face dataset cache.

**Spec:** `docs/superpowers/specs/2026-09-10-aijc-lighting-v2-design.md`

## Global Constraints

- Do not copy or directly ensemble any published ready-made test-label submission.
- Public architectures, pretrained weights and modeling code ideas may be studied and reimplemented.
- Illumination-changing augmentation is disabled in supervised CNN training because illumination is the target.
- A V2 recommendation must beat the verified V1 OOF accuracy of 0.518 before leaderboard upload.
- Persist probabilities in manifest order: 1500 train rows and 300 test rows, three classes.
- Structural/generator rules may affect test predictions only after validation on known train labels.

---

### Task 1: Hierarchy and constrained-assignment primitives

**Files:**
- Create: `aijc-lighting/src/hierarchical.py`
- Create: `aijc-lighting/src/transductive.py`
- Create: `aijc-lighting/tests/test_hierarchical_v2.py`
- Create: `aijc-lighting/tests/test_transductive_v2.py`

**Interfaces:**
- Produces: `compose_three_class_probs(p_dark: np.ndarray, p_bright_given_non_dark: np.ndarray) -> np.ndarray`
- Produces: `balanced_assignment(probs: np.ndarray, counts: tuple[int,int,int]) -> np.ndarray`
- Produces: `evaluate_count_constraint(probs: np.ndarray, y: np.ndarray) -> dict`

- [ ] **Step 1: Write failing hierarchy tests**

```python
import numpy as np
from src.hierarchical import compose_three_class_probs


def test_compose_three_class_probs_is_normalized():
    p_dark = np.array([0.8, 0.1])
    p_bright = np.array([0.25, 0.75])
    out = compose_three_class_probs(p_dark, p_bright)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.sum(1), 1.0)
    np.testing.assert_allclose(out[0], [0.8, 0.15, 0.05])
```

- [ ] **Step 2: Write failing constrained-assignment tests**

```python
import numpy as np
from src.transductive import balanced_assignment


def test_balanced_assignment_obeys_exact_counts():
    probs = np.array([
        [.9,.05,.05], [.8,.1,.1], [.1,.8,.1],
        [.1,.7,.2], [.1,.2,.7], [.1,.1,.8],
    ])
    pred = balanced_assignment(probs, (2, 2, 2))
    assert np.bincount(pred, minlength=3).tolist() == [2, 2, 2]
```

- [ ] **Step 3: Run tests and confirm import failures**

Run: `cd aijc-lighting && pytest tests/test_hierarchical_v2.py tests/test_transductive_v2.py -q`
Expected: FAIL because V2 modules do not exist.

- [ ] **Step 4: Implement probability composition and maximum-score count assignment**

Use clipped log probabilities and `scipy.optimize.linear_sum_assignment` against a repeated class-slot vector. `evaluate_count_constraint` compares unconstrained argmax accuracy to an assignment using the true validation class counts.

- [ ] **Step 5: Run the two new test files**

Run: `cd aijc-lighting && pytest tests/test_hierarchical_v2.py tests/test_transductive_v2.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat(aijc-lighting): add hierarchy and constrained assignment primitives`

---

### Task 2: Advanced illumination features and hierarchical CV

**Files:**
- Create: `aijc-lighting/src/features_v2.py`
- Modify: `aijc-lighting/src/hierarchical.py`
- Create: `aijc-lighting/tests/test_features_v2.py`

**Interfaces:**
- Produces: `extract_v2_features(path) -> dict[str, float]`
- Produces: `extract_v2_feature_frame(paths) -> pd.DataFrame`
- Produces: `run_hierarchical_cv(X, X_test, y, n_splits=5, seed=42) -> CVResult-compatible dict`

- [ ] **Step 1: Add a synthetic-image feature test**

Create black, gray and clipped-white images and assert feature finiteness, stable column order, and monotonicity of core exposure statistics.

- [ ] **Step 2: Run the new feature test and confirm failure**

Run: `cd aijc-lighting && pytest tests/test_features_v2.py -q`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement V2 features**

Compute multi-threshold shadow/highlight fractions, luminance percentiles, HSV/Lab moments, chromaticity `(r/(r+g+b), g/(...), b/(...))`, channel log-ratios, local 4x4 exposure histograms, top/bottom/center/border statistics, local contrast percentiles, Sobel/Laplacian visibility, dark-region and bright-region noise estimates, Retinex residual summaries, FFT radial-band energy, DCT low-frequency coefficients, and gamma-response summaries.

- [ ] **Step 4: Implement hierarchical CV**

Stage 1 evaluates CatBoost, LightGBM and ExtraTrees on `dark` versus non-dark. Stage 2 trains only rows with labels 1/2 and evaluates CatBoost, LightGBM, ExtraTrees and RBF-SVM. Select stage members by binary OOF accuracy and reconstruct three-class OOF/test probabilities with `compose_three_class_probs`.

- [ ] **Step 5: Run all unit tests**

Run: `cd aijc-lighting && pytest tests -q`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

Commit message: `feat(aijc-lighting): add v2 illumination features and hierarchical CV`

---

### Task 3: Foundation-backbone experiment runner

**Files:**
- Create: `aijc-lighting/run_v2.py`
- Modify: `aijc-lighting/src/solutions.py`
- Create: `aijc-lighting/tests/test_run_v2_config.py`

**Interfaces:**
- CLI: `python run_v2.py --track hierarchy|embedding|cnn|audit --backbone <name> --output-dir <path>`
- Embedding backbones: `vit_small_patch14_dinov2.lvd142m`, `convnext_tiny.fb_in22k_ft_in1k`, `efficientnet_b0.ra_in1k` with torchvision fallbacks.

- [ ] **Step 1: Write parser/backbone configuration tests**

Assert the runner accepts each track and rejects unknown tracks before touching the dataset.

- [ ] **Step 2: Run parser tests and confirm failure**

Run: `cd aijc-lighting && pytest tests/test_run_v2_config.py -q`
Expected: FAIL because `run_v2.py` does not exist.

- [ ] **Step 3: Extend embedding evaluation**

For each backbone evaluate embedding-only and embedding+V2-physics matrices with LogisticRegression, linear SVM, PCA(<=256)+RBF-SVM, LightGBM and CatBoost. Save `oof_probs.npy`, `test_probs.npy`, `metrics.json`, and `submission.csv`.

- [ ] **Step 4: Add reusable CNN configurations**

Support torchvision/timm model creation for EfficientNet-B0, ResNet18 and ConvNeXt-Tiny, geometry-only augmentation, progressive unfreezing and MPS/CUDA/CPU device selection.

- [ ] **Step 5: Run all tests**

Run: `cd aijc-lighting && pytest tests -q`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat(aijc-lighting): add configurable v2 backbone runner`

---

### Task 4: Deep generator and scene audit

**Files:**
- Create: `aijc-lighting/scripts/audit_scenes.py`
- Modify: `aijc-lighting/src/leakage.py`
- Create: `aijc-lighting/tests/test_scene_audit.py`

**Interfaces:**
- Produces: `normalized_scene_descriptor(path) -> np.ndarray`
- Produces: `scene_neighbor_report(train_paths, test_paths, labels, embeddings=None) -> dict`

- [ ] **Step 1: Add descriptor invariance test**

Generate one synthetic scene and gamma/brightness variants; assert exposure-normalized descriptors are closer to one another than to an unrelated synthetic scene.

- [ ] **Step 2: Run test and confirm failure**

Run: `cd aijc-lighting && pytest tests/test_scene_audit.py -q`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement normalized scene descriptors and metadata audit**

Equalize/standardize grayscale structure before downsampling; combine low-frequency DCT, gradient orientation histograms and normalized block means. Parse PNG chunks and report chunk names/lengths, gamma/ICC presence, compression size and ZIP member order when the source archive is available.

- [ ] **Step 4: Add nearest-neighbor label-consistency diagnostics**

Measure train leave-one-out nearest-neighbor label accuracy and cross-class extremely-close pairs for normalized descriptors and optional foundation embeddings. Report evidence but do not automatically alter labels.

- [ ] **Step 5: Run tests and commit**

Run: `cd aijc-lighting && pytest tests -q`
Expected: PASS.
Commit: `feat(aijc-lighting): deepen scene and generator audit`

---

### Task 5: OOF aggregation, repeated-CV checks and count constraints

**Files:**
- Create: `aijc-lighting/scripts/aggregate_v2.py`
- Modify: `aijc-lighting/src/ensemble.py`
- Create: `aijc-lighting/tests/test_aggregate_v2.py`

**Interfaces:**
- CLI: `python scripts/aggregate_v2.py --inputs outputs-v2/* --train-csv data/train.csv --test-csv data/test.csv --output-dir outputs-v2/final`

- [ ] **Step 1: Add aggregation test with synthetic probability files**

Assert candidate discovery, convex blend search, exact test row order and optional balanced assignment behavior.

- [ ] **Step 2: Confirm test failure, then implement aggregator**

Search single models, pair/triple sparse convex blends, and regularized multinomial stacking on OOF predictions. Report unconstrained and validation-count-constrained OOF accuracy. Enable `100/100/100` test assignment only if count-constrained OOF improves across the available validation results.

- [ ] **Step 3: Run all tests**

Run: `cd aijc-lighting && pytest tests -q`
Expected: PASS.

- [ ] **Step 4: Commit**

Commit message: `feat(aijc-lighting): add v2 OOF aggregation and constraints`

---

### Task 6: Parallel GitHub Actions matrix and real-data iteration

**Files:**
- Create: `.github/workflows/aijc-lighting-v2-matrix.yml`
- Modify: `aijc-lighting/README.md`

**Interfaces:**
- Matrix jobs: hierarchy, DINOv2, ConvNeXt embedding, EfficientNet embedding, CNN EfficientNet, CNN ResNet18, CNN ConvNeXt, audit.
- Aggregation job consumes probability artifacts and emits `submission_v2_best.csv`.

- [ ] **Step 1: Create workflow with one experiment per matrix job**

Restore the private HF dataset archive, extract it, install dependencies, run the selected track, upload compact probability/metric artifacts, and use caches keyed by backbone.

- [ ] **Step 2: Use compute appropriate to each job**

Default CPU/tabular jobs to Ubuntu. Add a separate macOS Apple-Silicon CNN experiment only as a measured alternative; never assume MPS is faster without timing. Remove the previous global `OMP_NUM_THREADS=2`/`MKL_NUM_THREADS=2` bottleneck for tabular jobs.

- [ ] **Step 3: Run CI and inspect each completed metric**

Reject failed/underperforming branches without blending them. Keep iterating on the strongest normal-vs-bright and embedding/CNN branches when they show OOF gains.

- [ ] **Step 4: Aggregate and verify final artifact**

Verify `submission_v2_best.csv` has columns `id,label`, exactly 300 unique IDs in official test order, integer labels in `{0,1,2}`, and is derived from the highest validated OOF configuration.

- [ ] **Step 5: Update README and commit**

Record measured V1 and V2 scores, chosen model, class distribution, runtime and exact reproduction command.
Commit: `ci(aijc-lighting): add parallel v2 max-score experiments`
