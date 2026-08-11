# E-CUP v5 Finalization and ≥0.6000 Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified six-signal submission and an honestly cross-fitted validation recipe with Macro AP ≥0.6000, while adding progress/resource telemetry to long-running stages.

**Architecture:** First repair artifact-layout resolution and freeze the existing 0.5975445721 v5 without retraining. Then add reusable progress telemetry. Finally search low-capacity grouped meta-blends over saved six-signal OOF predictions; use the RTX 2060 only if the meta-blend gate is not met. The selected recipe is refit for production and packaged with exact organizer-image smoke testing.

**Tech Stack:** Python 3.11, NumPy, pandas, scikit-learn, PyArrow, PyTorch/Transformers, GitHub Actions, self-hosted CUDA runner, Docker.

## Global Constraints

- Preserve all six v5 signals: `weak`, `sparse`, `explicit`, `contrastive`, `teacher`, `typed_explicit`.
- Sealed gold labels remain excluded from training and model selection.
- Validation headline must be fully out-of-fold at both base-model and meta-blend levels.
- Do not trade away model components because local hardware is slower than organizer hardware.
- Final ZIP must pass `odsai/ecup26-matching-baseline:1.0` with `--network none`.

---

### Task 1: Robust artifact layout resolution

**Files:**
- Create: `ecup_matching/ci/__init__.py`
- Create: `ecup_matching/ci/v5_artifacts.py`
- Create: `ecup_matching/tests/test_v5_artifacts.py`
- Modify: `.github/workflows/ecup-v5-package-first-structured.yml`

**Interfaces:**
- Produces: `resolve_structured_artifact(root: Path) -> StructuredArtifactPaths` with `.model` and `.legacy_runtime` paths.
- Consumes: extracted GitHub artifact directory.

- [ ] Write a failing test with the real macOS artifact layout: `out/model_v5_structured.joblib` plus `legacy/legacy_ecup/ml/...`.
- [ ] Run the focused test in Actions and confirm it fails because the resolver does not exist.
- [ ] Implement the minimal resolver supporting both current root layout and common-parent macOS layout; fail with an inventory of discovered files.
- [ ] Update packaging workflow to call the resolver and use returned paths instead of hard-coded root paths.
- [ ] Run focused tests, then the full test suite.
- [ ] Re-run package workflow and verify artifact download → tests → ZIP → organizer smoke → upload all succeed.

### Task 2: Observable progress telemetry

**Files:**
- Create: `ecup_matching/ml/progress.py`
- Create: `ecup_matching/tests/test_progress.py`
- Modify: `ecup_matching/ml/train_v5_production_structured.py`
- Modify: `ecup_matching/ml/train_v5_production_contrastive.py`
- Modify: `ecup_matching/ml/train_v5_production_teacher.py`
- Modify: `.github/workflows/ecup-v5-build-best-submit.yml`
- Modify: relevant v5 GPU workflow in private dispatcher after the public helper is stable.

**Interfaces:**
- Produces: `ProgressTracker(phase, total=None, report_every=...)`, `.update(done=None, increment=None)`, `.finish()`, and JSON-serializable snapshots.

- [ ] Write failing tests for percent, throughput, ETA behavior, finish state, and JSON serialization with an injected clock.
- [ ] Implement CPU RSS/peak RSS reporting and optional CUDA allocated/reserved memory reporting.
- [ ] Instrument structured phases (base features, weak sampling/features, sparse, explicit, typed explicit), neural steps, and packaging phases.
- [ ] Persist a timing JSON and add a concise GitHub step-summary table.
- [ ] Run the full test suite.

### Task 3: Strict grouped meta-blend experiment

**Files:**
- Create: `ecup_matching/ml/v5_meta_blend.py`
- Create: `ecup_matching/ml/run_v5_meta_blend.py`
- Create: `ecup_matching/tests/test_v5_meta_blend.py`
- Create/modify: a GitHub Actions experiment workflow that downloads the saved six-signal OOF predictions and sealed split metadata.

**Interfaces:**
- Consumes: six aligned OOF score vectors, target, category, item/component grouping.
- Produces: fully meta-OOF predictions, selected candidate description, global/category parameters, per-category AP, aggregate Macro AP.

- [ ] Write failing tests proving no meta-test row/group participates in its fold's weight fit.
- [ ] Implement global non-negative simplex rank weights with deterministic coordinate/grid optimization on meta-train only.
- [ ] Implement regularized category deviations toward global weights with explicit shrinkage hyperparameters selected inside training folds only.
- [ ] Add strongly regularized logistic/rank stacker as a comparator.
- [ ] Run grouped meta-cross-validation, save predictions and metrics, and accept only metrics calculated from held-out meta-fold predictions.
- [ ] Stop meta-search when a reproducible candidate reaches ≥0.6000; otherwise proceed to Task 4.

### Task 4: RTX neural improvement fallback

**Files:**
- Modify/add trusted v5 profile in private `MakSoS1/gpu-dispatch`.
- Reuse existing public neural trainers; add only configuration/progress hooks needed for the selected experiment.

**Interfaces:**
- Consumes: sealed split, pinned model revision, allowed development/weak rows.
- Produces: candidate OOF predictions and a production checkpoint only after validation acceptance.

- [ ] Validate runner labels and CUDA/VRAM at job start.
- [ ] Run one hypothesis at a time: improved contrastive curriculum or teacher training configuration, preserving objective/data leakage constraints.
- [ ] Generate OOF predictions for the changed signal and rerun the same grouped meta-blend evaluation.
- [ ] Accept only a reproducible strict Macro AP ≥0.6000 and record delta by category.

### Task 5: Production freeze and final submission

**Files:**
- Modify: `ecup_matching/ml/v5_production.py` if final fusion differs from equal-rank.
- Modify: `ecup_matching/submission/predict_v5.py` to apply the selected frozen fusion parameters.
- Modify: `ecup_matching/build_submission_v5.py` for provenance/manifest inclusion if needed.
- Create/update: `ecup_matching/experiments/v5/RESULTS.md`, `CURRENT.json`/project state files already used by the repository.

**Interfaces:**
- Produces: immutable final submission ZIP and metrics/provenance JSON.

- [ ] Add a failing unit test asserting production fusion exactly reproduces saved meta recipe on synthetic aligned inputs.
- [ ] Implement the selected frozen fusion with no target access at inference.
- [ ] Refit permitted production parameters on all development OOF rows while retaining strict meta-OOF score as the headline metric.
- [ ] Run full tests and exact organizer-image offline smoke.
- [ ] Freeze artifact by SHA-256, record source commit, split SHA, component artifact digests, validation metric, and ZIP size.
- [ ] Download the final artifact and provide the submission ZIP to the user.
