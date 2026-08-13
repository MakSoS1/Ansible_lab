# E-CUP v11 FastLex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new submission architecture with no neural encoder at inference, strict component-disjoint OOF competitive with v7/v10, and a large runtime margin on the full 275k fixture.

**Architecture:** Replace Transformer inference and SequenceMatcher-heavy fuzzy passes with vectorized lexical/sparse similarities, typed numeric/model/quantity/attribute consistency, category-aware nonlinear stacking, and cheap graph rescoring. Train/evaluate under the existing immutable component-disjoint folds and package only after exact organizer-image full-size runtime verification.

**Tech Stack:** Python, pandas, NumPy, SciPy sparse, scikit-learn HashingVectorizer/TfidfTransformer/HistGradientBoostingClassifier, joblib.

## Global Constraints

- Do not open sealed gold.
- Do not use Transformer/PyTorch/CUDA in submission runtime.
- Do not use difflib.SequenceMatcher in v11 runtime.
- Preserve organizer output contract `id1,id2,predict`.
- Use the existing immutable five component-disjoint folds and organizer Macro AP.
- Retain only candidates with measured 275k organizer-image wall time <= 90 seconds on the RTX host; target substantially below this.
- Record leaderboard separately from OOF; never relabel local OOF as leaderboard evidence.

---

### Task 1: Fast lexical feature kernel

**Files:**
- Create: `ecup_matching/ml/v11_fastlex.py`
- Create: `ecup_matching/tests/test_v11_fastlex.py`

**Interfaces:**
- Produces `build_item_cache(items: pd.DataFrame) -> dict`
- Produces `build_fast_pair_features(items, pairs, cache=None) -> pd.DataFrame`

- [ ] Write failing tests for normalization, exact/contains, token/char ngram overlap, number/model/quantity conflicts, attribute agreement, deterministic row order, and absence of SequenceMatcher.
- [ ] Run the targeted test and verify RED.
- [ ] Implement a vectorized/set-based kernel with no quadratic scans and no fuzzy edit distance.
- [ ] Run targeted tests and verify GREEN.

### Task 2: Sparse semantic channel

**Files:**
- Create: `ecup_matching/ml/v11_sparse.py`
- Create: `ecup_matching/tests/test_v11_sparse.py`

**Interfaces:**
- Produces `fit_sparse_bundle(train_items) -> dict`
- Produces `sparse_pair_scores(bundle, items, pairs) -> np.ndarray`

- [ ] Write failing tests for deterministic char+word hashing, cosine range, identical-text dominance, and bounded memory.
- [ ] Run RED.
- [ ] Implement HashingVectorizer-based item vectors and direct pair cosine without vocabulary fitting at inference.
- [ ] Run GREEN.

### Task 3: Leakage-safe v11 OOF stack

**Files:**
- Create: `ecup_matching/ml/run_v11_outer_oof.py`
- Create: `ecup_matching/ml/v11_stack.py`
- Create: `ecup_matching/tests/test_v11_stack.py`

**Interfaces:**
- Consumes fast lexical features and sparse score.
- Produces fold-local predictions and summary JSON.

- [ ] Write leakage tests proving held-fold labels cannot affect held predictions.
- [ ] Run RED.
- [ ] Fit category-balanced HGB plus category-shrunk calibration only on outer-train rows.
- [ ] Add target-free graph rescoring as a separately measured delta.
- [ ] Run all v11 stack tests GREEN.
- [ ] Execute full five-fold OOF and record per-fold/category metrics.

### Task 4: Production runtime

**Files:**
- Create: `ecup_matching/submission/predict_v11.py`
- Create: `ecup_matching/submission/build_submission_v11.py`
- Create: `ecup_matching/tests/test_predict_v11.py`

**Interfaces:**
- Produces `predict_to_csv_v11(...)`.

- [ ] Write organizer-contract and no-neural-import tests first.
- [ ] Run RED.
- [ ] Implement inference using only pandas/NumPy/SciPy/scikit-learn/joblib runtime closure.
- [ ] Run GREEN and full repository tests.

### Task 5: Exact full-size runtime gate and publication

**Files:**
- Create: `.github/workflows/ecup-v11-fastlex-oof.yml`
- Create: `.github/workflows/ecup-v11-fastlex-build.yml`
- Create/update private GPU workflow for exact 275k gate.
- Update: `ecup_matching/experiments/v11/RESULTS.md`, `SAFE_METRICS.json`, `CURRENT.json`, Memora state.

- [ ] Run exact organizer-image 275k fixture with full container wall timing.
- [ ] Reject any candidate above 90 seconds or with invalid/nonconstant output.
- [ ] Compare v11 strict OOF to v10 and historical v7/v9 evidence.
- [ ] Package deterministic ZIP only after quality/runtime gates pass.
- [ ] Publish exact keeper ZIP plus manifest to private Hugging Face.
- [ ] Run full repository verification and Memora checkpoint.
