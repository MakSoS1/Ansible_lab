# E-CUP v6 Fast Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an organizer-compatible E-CUP Matching v6 submission with honest strict component-disjoint OOF Macro AP >= 0.6000 and materially lower end-to-end inference time than v5, verified on the connected RTX 2060 SUPER runner and packaged as a final ZIP.

**Architecture:** Start from the retained v5 0.6018115534 six-signal meta ensemble. First profile the exact v5 ZIP on the GPU runner to measure phase costs. Then optimize the neural paths without changing score semantics where possible: batched text serialization/tokenization, CUDA mixed precision only when numerically safe, pinned/non-blocking transfers where supported, and reduced Python/Pandas overhead. If the exact-score path cannot meet the runtime gate, evaluate a predeclared fast architecture that removes or distills the expensive signal(s), and retain only candidates whose strict 5-fold component-disjoint OOF is >= 0.6000. Final packaging must run in `odsai/ecup26-matching-baseline:1.0`, offline, read-only, with CUDA enabled.

**Tech Stack:** Python 3.11, PyTorch/Transformers, scikit-learn 1.9.0, pandas/NumPy, GitHub Actions, self-hosted NVIDIA RTX 2060 SUPER, organizer Docker image `odsai/ecup26-matching-baseline:1.0`.

## Global Constraints

- Strict local selection metric: unweighted mean of `sklearn.metrics.average_precision_score` across exactly 20 official categories.
- Validation remains the immutable component/item-disjoint five-fold development split with SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed gold stays unopened and contributes zero rows to model selection.
- Retained v6 candidate must have strict OOF Macro AP >= 0.6000.
- Runtime is a first-class gate; benchmark exact submission code in the organizer image with CUDA on `ecup-rtx2060`.
- No network access during final inference.
- Final output columns must be exactly `id1,id2,predict`, preserve pair order, contain every row, finite scores, and scores in [0,1].
- Update `ecup_matching/experiments/v6/RESULTS.md`, `SAFE_METRICS.json`, canonical agent-memory files, and submission provenance.

---

### Task 1: Profile exact v5 runtime on RTX 2060

**Files:**
- Create in private dispatcher: `.github/workflows/ecup-v6-profile-v5.yml`
- No production-model changes in this task.

**Interfaces:**
- Consumes: exact retained v5 submission artifact `9116032675`, competition parquet files already cached under `/srv/github-gpu/data`, organizer image.
- Produces: timestamped end-to-end timings for 64, 2,000, and 10,000 pair samples and CUDA proof.

- [ ] **Step 1: Add a diagnostic workflow that runs the exact v5 ZIP in the organizer image on CUDA.**
- [ ] **Step 2: Run the workflow and confirm `torch.cuda.is_available()` plus the NVIDIA device name.**
- [ ] **Step 3: Record log timestamps around structured, contrastive, teacher, fusion, and CSV phases; if current runtime does not expose phase timings, infer coarse phase timing from timestamped progress and add temporary diagnostic instrumentation only in the private dispatcher copy.**
- [ ] **Step 4: Write the profile evidence into `ecup_matching/experiments/v6/RESULTS.md` before selecting an optimization.**

### Task 2: Add test-first fast-runtime helpers

**Files:**
- Create: `ecup_matching/tests/test_v6_fast_inference.py`
- Create: `ecup_matching/submission/v6_fast.py`

**Interfaces:**
- Produces: `select_runtime_config(total_memory_bytes: int, device_type: str) -> RuntimeConfig`, `batch_index_ranges(row_count: int, batch_size: int)`, and phase-timing helpers used by v6 inference.

- [ ] **Step 1: Write failing tests for deterministic batch selection on CPU, 8 GiB CUDA, 24 GiB CUDA, and >=60 GiB CUDA.**
- [ ] **Step 2: Run only `test_v6_fast_inference.py` and verify RED because the module/functions do not exist.**
- [ ] **Step 3: Implement the minimal helpers. RTX 2060 uses FP16 autocast; CPU uses no autocast.**
- [ ] **Step 4: Run the test file and verify GREEN.**

### Task 3: Optimize exact-score neural inference without changing model semantics

**Files:**
- Modify: `ecup_matching/submission/predict_v5.py`
- Test: `ecup_matching/tests/test_v6_fast_inference.py`

**Interfaces:**
- Consumes: v5 contrastive and teacher model directories, legacy text normalization/serialization.
- Produces: the same six signal vectors and final meta score with phase timing instrumentation.

- [ ] **Step 1: Add failing tests around runtime-config usage, stable output shape, and CPU fallback for mocked tiny torch models.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Replace repeated DataFrame slicing/list conversion in neural loops with precomputed aligned ID arrays/text lists; use runtime-configured batches; enable CUDA autocast; use non-blocking transfers when CUDA; keep final score arrays float64.**
- [ ] **Step 4: Add phase timers for load/structured/contrastive/teacher/meta/write.**
- [ ] **Step 5: Verify unit tests and the full v5 production test subset.**

### Task 4: Benchmark exact-score v6 candidate on RTX 2060

**Files:**
- Create in private dispatcher: `.github/workflows/ecup-v6-benchmark.yml`

**Interfaces:**
- Consumes: `ecup-v6-fast` source commit and exact v5 model artifacts.
- Produces: timings, rows/s, peak GPU memory, and numerical comparison against v5 on the same sample.

- [ ] **Step 1: Run 10,000-pair organizer-image CUDA benchmark for v5 and v6 on the same rows.**
- [ ] **Step 2: Assert output schema/order and finite range.**
- [ ] **Step 3: Compare v5 and v6 predictions; exact-score optimization must preserve rank ordering to numerical tolerance.**
- [ ] **Step 4: If projected full-test runtime has adequate margin, retain exact-score v6 and skip Task 5. Otherwise continue to Task 5.**

### Task 5: Build a fast >=0.6000 fallback architecture if exact-score runtime is insufficient

**Files:**
- Create: `ecup_matching/ml/run_v6_fast_ablation.py`
- Create: `ecup_matching/tests/test_v6_fast_ablation.py`
- Create workflow: `.github/workflows/ecup-v6-fast-ablation.yml`

**Interfaces:**
- Consumes: existing fully outer-cross-fitted six-signal OOF vectors and official categories.
- Produces: predeclared fast candidate OOF metrics using only inference-cheaper subsets and fixed meta strategies.

- [ ] **Step 1: Write failing tests proving held-fold labels cannot affect held-fold predictions and that candidate signal sets are fixed before evaluation.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Evaluate a small predeclared candidate set ordered by expected runtime: all structured plus contrastive but no teacher; all structured plus teacher but no contrastive; and fixed cheap-signal meta variants. Do not search weights after inspecting results.**
- [ ] **Step 4: Keep only candidates with strict OOF >=0.6000 and choose the fastest by measured GPU runtime.**
- [ ] **Step 5: If none reaches 0.6000, retain the exact-score optimized architecture and continue optimizing its runtime rather than lowering the quality gate.**

### Task 6: Build and verify final v6 submission

**Files:**
- Create: `.github/workflows/ecup-v6-final-submit.yml`
- Create/update: `ecup_matching/experiments/v6/RESULTS.md`
- Create/update: `ecup_matching/experiments/v6/SAFE_METRICS.json`
- Update: `ecup_matching/experiments/CURRENT.json`
- Update: `docs/agent-memory/PROJECT_STATE.md`
- Update: `docs/agent-memory/EXPERIMENT_INDEX.md`
- Update: `docs/agent-memory/DECISIONS.md`

**Interfaces:**
- Produces: one final `ecup-v6-...-submission.zip`, SHA-256, runtime report, strict OOF evidence, private HF copy, and Actions artifact.

- [ ] **Step 1: Build the final ZIP from byte-verified retained v5 artifacts plus only the selected v6 runtime/meta changes.**
- [ ] **Step 2: Run organizer-image offline read-only CUDA smoke on RTX 2060.**
- [ ] **Step 3: Run a representative runtime benchmark and project/check against the competition timeout with safety margin.**
- [ ] **Step 4: Run the complete repository test suite.**
- [ ] **Step 5: Upload the exact smoked ZIP to private Hugging Face and GitHub Actions; record SHA-256 and artifact IDs.**
- [ ] **Step 6: Update all v6 documentation and canonical memory pointers with metric and runtime evidence, clearly separating local OOF from leaderboard score.**

## Self-Review

- Spec coverage: metric >=0.6000, runtime profiling, GPU runner use, organizer-image verification, documentation, and final archive are all covered.
- Placeholder scan: no TBD/TODO/implement-later placeholders.
- Type consistency: v6 helper interfaces are defined once and reused by later tasks.
