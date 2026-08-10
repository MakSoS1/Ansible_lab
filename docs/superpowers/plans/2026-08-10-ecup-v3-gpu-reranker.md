# E-CUP v3 GPU Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and retain an E-CUP 2026 v3 compact neural reranker with model-mined hard negatives on free GPU compute, blend it with v2b, and produce a verified private submission ZIP that improves fixed item-disjoint Macro AP.

**Architecture:** GitHub Actions prepares a compact leakage-safe neural dataset and orchestration metadata from private HF. Preferred compute is a private Hugging Face ZeroGPU Gradio Space so the same HF account holds both data and outputs. A compact RuBERT-tiny2 sequence-classification model is trained in stage 1, mines false-positive human negatives, receives a short stage-2 fine-tune, then validation chooses neural/structured blend or gating. Final packaging reuses organizer-compatible v2 runtime and adds neural artifacts only if validation and runtime gates pass.

**Tech Stack:** Python 3.11/3.12, pandas/pyarrow, numpy, scikit-learn, PyTorch, Transformers, Hugging Face Hub/Spaces/Gradio, GitHub Actions, Docker organizer image, hardened Memora.

## Global Constraints

- Branch: `ecup-matching-2026`; never modify or merge `main`.
- Fixed human validation: 73,131 pairs with zero train/validation item overlap.
- Baseline to beat: v2b Macro AP `0.5010008994958702`.
- Raw data, derived pair text, weights and ZIPs are private HF artifacts only.
- No credentials in Git, logs, Memora or uploaded public artifacts.
- No paid compute; use only clearly free quota/credits.
- Final organizer benchmark must leave at least 25% headroom to 780 s.
- A failed/non-improving neural experiment is recorded as rejected/blocked, not relabeled as success.

---

### Task 1: Declare v3 and add experiment protocol files

**Files:**
- Modify: `ecup_matching/experiments/CURRENT.json`
- Create: `ecup_matching/experiments/v3/PLAN.md`
- Create: `ecup_matching/experiments/v3/RESULTS.md`

**Interfaces:**
- Consumes: v2 state from `CURRENT.json`, `PROJECT_STATE.md`, `EXPERIMENT_INDEX.md`.
- Produces: canonical v3 identity for workflows and Memora.

- [ ] **Step 1:** Set `CURRENT.json` to `version=v3`, `status=in_progress`, previous `v2`, plan/results paths, baseline score and expected private artifact path.
- [ ] **Step 2:** Write v3 `PLAN.md` with the exact fixed validation, GPU backend order, compact row budget, model, hard-negative rules, metric/runtime gates and abort criteria.
- [ ] **Step 3:** Create `RESULTS.md` with status `in progress` and placeholders expressed as explicit pending fields rather than fake metrics.
- [ ] **Step 4:** Run `python scripts/memory_policy.py` in CI; expected PASS for an in-progress v3.
- [ ] **Step 5:** Commit as `docs: start E-CUP v3 GPU reranker`.

### Task 2: Compact leakage-safe neural dataset builder (TDD)

**Files:**
- Create: `ecup_matching/ml/v3_dataset.py`
- Create: `ecup_matching/tests/test_v3_dataset.py`
- Reuse: `ecup_matching/ml/train_reranker_v2.py`, `reranker_data.py`, `v2_split.py`, `weak_labels.py`

**Interfaces:**
- Produces: `build_v3_examples(...) -> (train_examples, validation_examples, report)` where report includes validation overlap, per-category counts and source counts.

- [ ] **Step 1: Write failing tests** covering deterministic sampling, zero validation item leakage, preservation of all validation rows, and minimum weak-category representation.
- [ ] **Step 2: Run RED** with `python -m pytest ecup_matching/tests/test_v3_dataset.py -q`; expected failure because `v3_dataset` does not exist.
- [ ] **Step 3: Implement minimal builder** using the existing fixed split/serializer. Human positives are retained, negatives/weak rows are stratified and weak categories may receive an explicit multiplier. Never sample validation.
- [ ] **Step 4: Run GREEN** for the new tests and then the full repository suite.
- [ ] **Step 5:** Commit as `feat: add leakage-safe v3 neural dataset builder`.

### Task 3: Model-mined hard-negative selector and blend search (TDD)

**Files:**
- Create: `ecup_matching/ml/v3_selection.py`
- Create: `ecup_matching/tests/test_v3_selection.py`

**Interfaces:**
- Produces: `select_hard_negatives(frame, scores, count, priority_categories, seed)` and `select_best_blend(structured, neural, target, category)`.

- [ ] **Step 1: Write failing tests** asserting only target=0 rows are mined, highest neural false positives are preferred per category, output is deterministic, priority categories receive quota, and blend search returns scores in `[0,1]`.
- [ ] **Step 2: Run RED** and confirm missing implementation.
- [ ] **Step 3: Implement selector** with per-category top-k then global fill; pair hard negatives with positives only in the training script, not selector.
- [ ] **Step 4: Implement blend sweep** over fixed alphas plus category-aware alpha search with v2 fallback.
- [ ] **Step 5: Run GREEN** and full suite.
- [ ] **Step 6:** Commit as `feat: add v3 hard-negative and blend selection`.

### Task 4: ZeroGPU trainer

**Files:**
- Create: `ecup_matching/ml/train_v3_reranker.py`
- Create: `ecup_matching/tests/test_train_v3_contract.py`
- Create: `ecup_matching/zerogpu/app.py`
- Create: `ecup_matching/zerogpu/requirements.txt`
- Create: `ecup_matching/zerogpu/README.md`

**Interfaces:**
- GPU worker consumes private derived parquet paths and writes `metrics.json`, `validation_predictions.parquet`, `model/`, tokenizer and manifest to private HF prefix `experiments/v3/neural/<run-id>/`.

- [ ] **Step 1: Write contract tests** for CLI arguments, CUDA requirement, explicit time/row budgets, private-only HF repo ID passed by environment, and secret-free logging surface.
- [ ] **Step 2: Run RED.**
- [ ] **Step 3: Implement trainer** by refactoring reusable functions from `train_reranker_v2.py`: stage-1 short training, full fixed validation scoring, model-mined hard negatives, short stage-2 fine-tune, retain better stage.
- [ ] **Step 4: Implement ZeroGPU app** with `@spaces.GPU(duration=300)`; root process downloads/tokenizer-prepares private derived inputs without putting the HF token in output; GPU function performs only training/scoring/upload.
- [ ] **Step 5: Run GREEN** and full suite.
- [ ] **Step 6:** Commit as `feat: add ZeroGPU v3 reranker trainer`.

### Task 5: Private derived-data preparation workflow

**Files:**
- Create: `.github/workflows/ecup-v3-prepare.yml`
- Create: `ecup_matching/v3_prepare.py`
- Create: `ecup_matching/tests/test_v3_prepare.py`

**Interfaces:**
- Reads private `items_human.parquet`, `matches.parquet`, `matches_llm.parquet`, `items.parquet`, v2 validation/structured artifacts.
- Writes only private HF `experiments/v3/prepared/<commit>/train_examples.parquet`, `validation_examples.parquet`, `structured_validation_predictions.parquet`, `manifest.json`.

- [ ] **Step 1: Write RED tests** for private path naming, artifact manifest and no secret/raw-data export.
- [ ] **Step 2: Implement preparation command** around Task 2 builder and retained v2 structured validation predictions.
- [ ] **Step 3: Workflow runs repository tests + memory policy, downloads private inputs with `HF_TOKEN`, prepares/upload derived files, verifies remote presence and cleans runner temp.
- [ ] **Step 4: Execute workflow** and record run/job IDs/data counts in v3 RESULTS.
- [ ] **Step 5:** Commit as `ci: prepare private v3 neural data`.

### Task 6: ZeroGPU Space orchestration and real training

**Files:**
- Create: `.github/workflows/ecup-v3-zerogpu.yml`
- Create: `ecup_matching/v3_zerogpu_orchestrate.py`
- Create: `ecup_matching/tests/test_v3_zerogpu_orchestrate.py`

**Interfaces:**
- Creates/updates a private HF Space `Maksim123321/ecup-2026-v3-reranker` with Gradio/ZeroGPU when allowed, injects HF token as a Space secret, triggers one training call, polls output in private dataset repo, then pauses/removes the Space secret.

- [ ] **Step 1: Write RED tests** using a fake `HfApi` proving the Space is private, hardware requested is ZeroGPU, secret values are never printed, and cleanup removes the secret.
- [ ] **Step 2: Implement orchestrator** with explicit failure states for free-account ZeroGPU unavailable/quota exhausted.
- [ ] **Step 3: Run GREEN** and full suite.
- [ ] **Step 4: Execute workflow.** If ZeroGPU is unavailable, record the exact platform error and proceed to the next automatable free backend only if secure credentials already exist; do not publish data or spend paid credits.
- [ ] **Step 5: Wait/poll until private metrics/model are verified** or platform returns a terminal blocker.
- [ ] **Step 6:** Commit orchestration fixes encountered through systematic debugging.

### Task 7: Evaluate neural + structured candidates

**Files:**
- Create: `ecup_matching/ml/evaluate_v3.py`
- Create: `ecup_matching/tests/test_evaluate_v3.py`
- Update: `ecup_matching/experiments/v3/RESULTS.md`

**Interfaces:**
- Consumes v2 structured validation scores and ZeroGPU validation predictions.
- Produces `v3-validation-metrics.json`, selected blend/gate config and all 20 category APs.

- [ ] **Step 1: RED tests** for row/id alignment, same fixed validation target/category, and no silent merge mismatch.
- [ ] **Step 2: Implement evaluator** using exact official macro AP helper and Task 3 blend search.
- [ ] **Step 3: Evaluate neural stage1/stage2/global blends/category-aware blends.**
- [ ] **Step 4: Select v3 only if strictly better than v2b.** Otherwise continue safe free-GPU retry/alternative rather than claiming success.
- [ ] **Step 5:** Record exact metrics and commit `exp: evaluate v3 neural reranker`.

### Task 8: Build organizer-compatible v3 submission

**Files:**
- Create/modify focused files under `ecup_matching/submission/v3/`
- Create: `ecup_matching/build_submission_v3.py`
- Create: `ecup_matching/tests/test_submission_v3.py`
- Create: `.github/workflows/ecup-build-v3-submit.yml`

**Interfaces:**
- Produces private `submissions/v3/ecup-v3-submission.zip` plus metrics/manifest.

- [ ] **Step 1: RED tests** for metadata/run contract, local model paths, offline behavior, score range/order and absence of forbidden files.
- [ ] **Step 2: Implement minimal runtime** reusing v2 normalization/structured inference and adding compact neural batch inference + selected blend/gate.
- [ ] **Step 3: Package only necessary model/tokenizer files.**
- [ ] **Step 4: Exact organizer-image 1k smoke with `--network none`.
- [ ] **Step 5: 275k offline benchmark.** Require <=585 s (25% headroom) and verify schema/order/range.
- [ ] **Step 6: Upload ZIP and package metrics to private HF and verify presence.
- [ ] **Step 7:** Commit `feat: package verified v3 submission`.

### Task 9: Finish documentation and durable memory

**Files:**
- Update: `ecup_matching/experiments/v3/RESULTS.md`
- Update: `ecup_matching/experiments/CURRENT.json`
- Update: `docs/agent-memory/EXPERIMENT_INDEX.md`
- Update: `docs/agent-memory/PROJECT_STATE.md`
- Update: `docs/agent-memory/DECISIONS.md`
- Update: `AGENTS.md`

**Interfaces:**
- Produces complete cross-agent handoff and private Memora v3 checkpoint.

- [ ] **Step 1:** Record exact GPU backend/GPU model, run IDs, row counts, stage metrics, hard-negative metrics, selected blend, all category APs, ZIP path/size and runtime.
- [ ] **Step 2:** Set `CURRENT.status=completed` only after private submission and benchmark pass.
- [ ] **Step 3:** Run full repository tests and `python scripts/memory_policy.py`.
- [ ] **Step 4:** Allow memory workflow to restore prior DB, ingest canonical sources, run SQLite integrity + secret scan and upload `agent-memory/checkpoints/<timestamp>-v3-<sha>/` plus `agent-memory/latest/`.
- [ ] **Step 5:** Verify memory workflow success and record final checkpoint ID.
- [ ] **Step 6:** Final security scan of changed public files for token-like strings and forbidden binary/raw artifacts.
