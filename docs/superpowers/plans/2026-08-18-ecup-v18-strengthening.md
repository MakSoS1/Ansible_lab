# E-CUP v18 Strengthening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and queue an orthogonal v18 training/data/validation ladder that can produce a stronger single-RuBERT submission without increasing production inference complexity.

**Architecture:** Add v18-only weak-quality, hard-mining, train-view augmentation, full-encoder and EMA mechanisms behind explicit candidate configuration. A generic probe produces comparable fold/weak-holdout metrics; a deterministic selector promotes only independently passing mechanisms into a combined scaled candidate. Hosted macOS validates correctness cheaply, while canonical training remains serialized on the private RTX runner.

**Tech Stack:** Python 3, pandas, NumPy, PyTorch, Transformers, pytest, GitHub Actions, Docker, parquet/CSV, existing E-CUP v7/v17 training/runtime code.

**Spec:** `docs/superpowers/specs/2026-08-18-ecup-v18-strengthening-design.md`

## Global Constraints

- Sealed gold remains unopened and unscored.
- Immutable split SHA is `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Human/weak item overlap must remain exactly zero.
- Production remains one RuBERT pair CrossEncoder with max_length 256.
- Candidate gates are fixed before reading candidate results.
- No new external LLM label is required for v18.
- Production package is not emitted unless fold/weak/per-category promotion gates and organizer Check all pass.

---

### Task 1: Continuous weak confidence and curriculum

**Files:**
- Create: `ecup_matching/ml/v18_weak_quality.py`
- Test: `tests/test_v18_weak_quality.py`

**Interfaces:**
- Produces: `continuous_weak_weight(probability: float, dead_zone: float=0.05, gamma: float=1.5) -> float`
- Produces: `prepare_weak_pairs_v18(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]`
- Produces: `split_weak_curriculum(frame: pd.DataFrame, high_margin: float=0.30) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]`

- [ ] Write tests asserting p=0/1 -> 1, p=0.5 -> 0, symmetry around 0.5, medium confidence retained at low weight, duplicates collapse to highest-confidence target, and high curriculum is a subset of broad.
- [ ] Run `pytest -q tests/test_v18_weak_quality.py` and verify import failure before implementation.
- [ ] Implement deterministic canonicalization using existing `canonicalize_pairs`, continuous weights, dead-zone removal, hard target, duplicate collapse, `_weak_margin`, and curriculum split.
- [ ] Run the test file and verify all cases pass.
- [ ] Commit the implementation and tests.

### Task 2: Deterministic training views

**Files:**
- Create: `ecup_matching/ml/v18_views.py`
- Test: `tests/test_v18_views.py`
- Modify: `ecup_matching/ml/v7_neural.py`

**Interfaces:**
- Produces: `augment_serialized_view(text: str, *, drop_residual: bool, drop_numeric: bool) -> str`
- Extends `train_pair_phase(..., pair_swap_probability: float=0.0, residual_dropout_probability: float=0.0, numeric_dropout_probability: float=0.0, ema_decay: float | None=None, return_ema_state: bool=False)` without changing default behavior.

- [ ] Add tests proving protected `[CAT]`, `[NAME]`, `[BRAND]`, `[MODEL]`, `[IDENTITY]` lines are never dropped, while requested `[RESIDUAL]`/`[NUMERIC]` lines are removed.
- [ ] Add a deterministic epoch/index decision helper test: same seed/epoch/index gives same view, changing epoch changes at least some decisions.
- [ ] Run the tests before code and verify failure.
- [ ] Implement helpers in `v18_views.py` using SHA-256-derived uniform values, avoiding Python hash randomization.
- [ ] Modify the in-function `PairDataset` in `v7_neural.train_pair_phase` to expose `epoch`; use deterministic decisions at `__getitem__`/collate time for pair swap and safe low-priority field dropout. Set `dataset.epoch = epoch` before each loader pass.
- [ ] Preserve exact legacy behavior when all three probabilities are zero.
- [ ] Run existing v7 tests plus `tests/test_v18_views.py`.
- [ ] Commit.

### Task 3: EMA and hard-example selection

**Files:**
- Create: `ecup_matching/ml/v18_hard_mining.py`
- Create: `ecup_matching/ml/v18_ema.py`
- Test: `tests/test_v18_hard_mining.py`
- Test: `tests/test_v18_ema.py`
- Modify: `ecup_matching/ml/v7_neural.py`

**Interfaces:**
- Produces: `select_disagreement_hard_examples(frame, predictions, max_rows, seed) -> tuple[pd.DataFrame, dict[str, object]]`
- Produces: `ExponentialMovingAverage(model, decay)` with `update()`, `copy_to()`, `state_dict()`.
- `train_pair_phase` updates EMA after optimizer steps when configured and can return the EMA tensor state through an optional mutable holder argument `ema_state_out: dict[str, object] | None=None`.

- [ ] Test hard-mining ranking by `abs(pred-target)*weak_weight`, deterministic category/class balancing, and no target replacement.
- [ ] Test EMA arithmetic on a two-parameter toy module and serialization/copy behavior.
- [ ] Verify both tests fail before implementation.
- [ ] Implement deterministic hard-mining with per-(category, hard_target) quota followed by global highest-disagreement fill.
- [ ] Implement EMA over trainable floating-point parameters only.
- [ ] Integrate EMA update after each successful optimizer step in `train_pair_phase`; defaults must remain unchanged.
- [ ] Run new and relevant existing tests.
- [ ] Commit.

### Task 4: Generic v18 probe and improved metrics

**Files:**
- Create: `ecup_matching/ml/run_v18_probe.py`
- Create: `ecup_matching/ml/v18_metrics.py`
- Test: `tests/test_v18_metrics.py`

**Interfaces:**
- CLI accepts `--candidate {control,q1-quality,q2-hard,q3-views,q4-full,q5-ema,combined}`, `--fold`, weak exposure knobs, and output directory.
- Produces `metrics.json`, `v18-fold-oof.parquet`, and `active-learning.csv`.
- Produces `worst_qualifying_category_delta(candidate_per_category, control_per_category, min_rows=200) -> dict`.

- [ ] Write metrics tests for qualifying-category filtering, worst regression, and robust score ordering.
- [ ] Verify tests fail.
- [ ] Implement `run_v18_probe.py` by reusing frozen manifest loading, v17 weak holdout, v7 serializer/model loader and macro AP. Do not call the recomputing split builder.
- [ ] Q1 uses v18 continuous weak preparation and two-stage confidence curriculum.
- [ ] Q2 performs phase-1 weak training, predicts a deterministic mining pool, calls `select_disagreement_hard_examples`, and uses the mined/mixed rows in phase 2 without altering targets.
- [ ] Q3 enables pair swap 0.5, residual dropout 0.15 and numeric dropout 0.05.
- [ ] Q4 uses last 12 layers, LR `8e-6`, physical batch 16/effective 32.
- [ ] Q5 enables EMA decay `0.999` and evaluates raw vs EMA state, retaining both metric values in evidence.
- [ ] `combined` accepts a JSON list of approved mechanism names and enables exactly that union.
- [ ] Export active-learning rows stratified by category/hard class with id1/id2, weak target, weight, prediction, disagreement and reason code; never use them as additional training labels in the same run.
- [ ] Run tests and `python -m py_compile` on all v18 modules.
- [ ] Commit.

### Task 5: Preregistered selector

**Files:**
- Create: `ecup_matching/ml/v18_select.py`
- Test: `tests/test_v18_select.py`

**Interfaces:**
- Produces `evaluate_single(control: dict, candidate: dict) -> dict`
- Produces `select_mechanisms(control: dict, candidates: dict[str, dict]) -> dict`
- Produces `evaluate_combination(control: dict, singles: dict, combined: dict) -> dict`
- Produces `evaluate_scaled_confirmation(fold0_control, fold0_candidate, fold1_control, fold1_candidate) -> dict`

- [ ] Encode exact gates from the spec in tests, including strict `> +0.003`/`> +0.005` boundaries with numerical tolerance and category regression >= -0.03.
- [ ] Verify boundary tests fail before implementation.
- [ ] Implement pure deterministic selection functions; no environment-dependent thresholds.
- [ ] Run tests.
- [ ] Commit.

### Task 6: M1/MPS hosted smoke lane

**Files:**
- Create: `tests/test_v18_training_smoke.py`
- Create: `.github/workflows/ecup-v18-m1-smoke.yml`
- Create: `ecup_matching/experiments/v18/M1_SMOKE.md`

**Interfaces:**
- Workflow triggers on changes under v18 modules/tests or a `v18_m1_smoke_job.json` trigger.
- Produces a small JSON artifact recording architecture, PyTorch version, `mps_built`, `mps_available`, selected device, and test result.

- [ ] Build an in-memory tiny torch classification module + dummy tokenizer test that calls the v18-enabled `train_pair_phase` for a few batches on `mps` when available, otherwise CPU.
- [ ] Add synthetic tests for Q1-Q5 preprocessing/configuration without private data.
- [ ] Configure `runs-on: macos-15`, install only required Python wheels, print `uname -m` and MPS capability, run the v18 test subset.
- [ ] Trigger the workflow by committing `v18_m1_smoke_job.json` and record the run/result in `M1_SMOKE.md` after observation.
- [ ] Commit any execution-only compatibility fix without changing experimental gates.

### Task 7: RTX serialized candidate ladder

**Files in `MakSoS1/gpu-dispatch` branch `ecup-v18-executor`:**
- Create: `.github/workflows/ecup-v18-ladder.yml`
- Create: `v18_ladder_job.json`
- Create: `v18_run_ladder.py`

**Interfaces:**
- One self-hosted GPU job executes control/Q1-Q5 at historical exposure, then selector, then combined historical probe, then scaled combined fold0 and matching fold1 confirmation only if gates pass.
- Persistent outputs: `/srv/github-gpu/output/v18-ladder-${GITHUB_RUN_ID}/...`.

- [ ] Pin exact Ansible source SHA and canonical data/image/split invariants.
- [ ] Share `concurrency.group: ecup-isolated-gpu` with v17 so the v18 ladder queues behind current work and cannot overlap the only RTX.
- [ ] Run Q1-Q5 with identical fold0 weak-holdout seed and historical exposure `600000 x 0.35`.
- [ ] Invoke `v18_select` on result JSONs and write `selection.json` before combined run.
- [ ] If no mechanism passes, finish successfully with `no_keeper=true` and do not train a fabricated combination.
- [ ] Run combined at historical exposure; only on combination promotion run scaled `3000000/1500000/1.0` fold0 and fold1 confirmation.
- [ ] Store metrics/logs/active-learning outputs on persistent disk and upload small evidence artifacts with `continue-on-error` so quota cannot destroy the result.
- [ ] Trigger by committing `v18_ladder_job.json`.

### Task 8: Production refit and package

**Files in `MakSoS1/gpu-dispatch` branch `ecup-v18-executor`:**
- Create: `v18_build_final.py`
- Extend: `.github/workflows/ecup-v18-ladder.yml`

**Interfaces:**
- Production stage consumes only a scaled candidate with `evaluate_scaled_confirmation(...).promote == true`.
- Final file: `ecup-v18-strengthened-v7runtime-submission.zip`.

- [ ] Full-dev refit uses the exact selected v18 mechanisms, scaled weak exposure, frozen split loader, and no gold scoring.
- [ ] Bind checkpoint to `v18-training-policy.json` containing source SHA, selected mechanisms, all fold/weak/category gate evidence and data knobs.
- [ ] Build one-checkpoint v7-compatible ZIP using pinned packaging source.
- [ ] Audit ZIP paths, metadata, exactly one safetensors checkpoint, sealed-gold flags, and source/policy provenance.
- [ ] Run exact organizer image on a deterministic 1000-pair fixture with 60-second limit; validate pair order, finite predictions and >10 unique scores.
- [ ] Write manifest and SHA256SUMS only after the Check passes.
- [ ] Keep final bytes on persistent runner disk and attempt private Actions artifact export. Never publish model/data publicly.

### Task 9: Documentation and execution record

**Files:**
- Create/Update: `ecup_matching/experiments/v18/RESULTS.md`
- Update: `ecup_matching/experiments/CURRENT.json`
- Update: `docs/agent-memory/PROJECT_STATE.md` if present on the v18 source branch.

- [ ] Record every candidate, exact run ID, source SHA, weak exposure, fold metrics, weak metrics, category regression, selector decision, and failures.
- [ ] Clearly separate local promotion metrics from Public LB; never infer an unobserved leaderboard score.
- [ ] Record M1 smoke result and whether MPS was exposed by the hosted runner.
- [ ] If a final package exists, record filename, bytes, SHA-256 and organizer Check runtime.
- [ ] Commit documentation after measurable results are available.
