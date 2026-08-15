# E-CUP v15 Field-Aware Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a field-aware single-CrossEncoder product matcher, a fold-safe active-distillation pipeline, and an immutable GPU execution ladder that can produce strictly validated, runtime-safe submission candidates without disturbing ongoing v14 runs.

**Architecture:** `Ansible_lab` is the canonical research/source repository and owns the v15 experiment contract, tests, validation, memory and production packaging source. `gpu-dispatch` is the private execution plane: every workflow consumes an immutable job manifest tied to an exact `Ansible_lab` source SHA, runs in the trusted offline container, emits metrics/artifacts, and never becomes a competing source of architectural truth.

**Tech Stack:** Python 3, PyTorch, Hugging Face Transformers, pandas/pyarrow, scikit-learn `average_precision_score`, pytest, GitHub Actions, Docker, self-hosted RTX 2060 SUPER for training screens, organizer H100 as deployment target.

## Global Constraints

- Public target: Public Macro AP >= `0.50`; never present a local metric as a guaranteed leaderboard score.
- Official metric: unweighted mean of `average_precision_score` across all 20 categories.
- Canonical split: `365654` human rows = `285210` dev + `80444` sealed gold, 5 component-disjoint folds.
- Split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Canonical row-map SHA-256: `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`.
- Sealed gold remains unopened and unscored.
- Historical LLM/weak `target` is quarantined by default; weak parquet may be read as `id1,id2` candidate topology only.
- Final inference ships one tokenizer + one pair Transformer checkpoint + deterministic parser/fusion code; no teacher, second Transformer, HGB, TF-IDF, graph engine, network call or dynamic model download.
- Existing v14/A17 and other already-running/queued jobs are not cancelled, modified or reused as v15 evidence.
- Fold0 is a screen, not a keeper. Strict five-fold OOF is mandatory for a final v15 quality claim.
- Internal Check runtime target is `<=50s` on exact final bytes, preserving >=10s headroom under the organizer's 60s Check limit.

---

## File Structure

### Canonical public repository: `MakSoS1/Ansible_lab`

Create/modify:

- `docs/superpowers/specs/2026-08-16-ecup-v15-field-aware-distillation-design.md` — approved architecture design.
- `docs/superpowers/plans/2026-08-16-ecup-v15-field-aware-distillation.md` — this execution plan.
- `ecup_matching/experiments/v15/PLAN.md` — concise canonical experiment contract used by agents/Memora.
- `ecup_matching/experiments/v15/RESULTS.md` — append-only retained KEEP/REJECT/FAIL evidence.
- `ecup_matching/experiments/v15/SAFE_METRICS.json` — machine-readable safe aggregate metrics only.
- `ecup_matching/experiments/v15/MANIFEST.json` — frozen experiment identity and invariants.
- `ecup_matching/experiments/CURRENT.json` — current primary research stage; v14/A17 recorded as parallel legacy completion, not cancelled.
- `docs/agent-memory/PROJECT_STATE.md` — current external anchors, v14 evidence, v15 immediate action.
- `docs/agent-memory/EXPERIMENT_INDEX.md` — v13 actual LB, v14 causal ladder, v15 stage.
- `docs/agent-memory/DECISIONS.md` — durable D045+ decisions for v14/v15 architecture pivot and repo roles.
- `AGENTS.md` — current handoff and mandatory reading order updated to v15.
- `ecup_matching/v15_fields.py` — deterministic item-field parsing/normalization.
- `ecup_matching/v15_pair_features.py` — typed symmetric pair evidence.
- `ecup_matching/v15_serialization.py` — field-aware pair text construction.
- `ecup_matching/v15_model.py` — single Transformer + optional typed-feature/category residual head.
- `ecup_matching/v15_train.py` — fold-safe A0-A4 training entry point.
- `ecup_matching/v15_validate.py` — canonical fold selection, Macro AP, 20-category and hard-slice diagnostics.
- `ecup_matching/v15_distill.py` — B0-B3 teacher/student data contracts; legacy target must not be read in B1.
- `ecup_matching/v15_runtime.py` — referenced-item materialization, parse cache, batched offline inference.
- `ecup_matching/v15_build_submission.py` — deterministic one-model package builder.
- `ecup_matching/tests/test_v15_fields.py`
- `ecup_matching/tests/test_v15_pair_features.py`
- `ecup_matching/tests/test_v15_serialization.py`
- `ecup_matching/tests/test_v15_model.py`
- `ecup_matching/tests/test_v15_validation.py`
- `ecup_matching/tests/test_v15_distill.py`
- `ecup_matching/tests/test_v15_runtime.py`

### Private executor: `MakSoS1/gpu-dispatch`

Create/modify:

- `docs/v15-executor-contract.md` — private executor rules pointing to the canonical public plan.
- `v15_job_contract.py` — schema/provenance validator for immutable job manifests.
- `tests/test_v15_job_contract.py` — fail-closed manifest tests.
- `v15_a0_job.json`, `v15_a1_job.json`, `v15_a2_job.json` — fold0 screen manifests bound to exact public source SHA.
- `.github/workflows/ecup-v15-a0.yml`, `.github/workflows/ecup-v15-a1.yml`, `.github/workflows/ecup-v15-a2.yml` — sequential GPU screens.
- `.github/workflows/ecup-v15-teacher-b0.yml` — conditional teacher screen, never required before A-family quality evidence.
- `.github/workflows/ecup-v15-strict.yml` — manual/manifest-gated folds 1-4 only after a strong fold0 candidate.
- `.github/workflows/ecup-v15-production.yml` — manual/manifest-gated full-dev refit/package/runtime; must refuse unpromoted configs.

---

### Task 1: Canonical v15 memory and experiment contract

**Files:**
- Create: `ecup_matching/experiments/v15/PLAN.md`
- Create: `ecup_matching/experiments/v15/RESULTS.md`
- Create: `ecup_matching/experiments/v15/SAFE_METRICS.json`
- Create: `ecup_matching/experiments/v15/MANIFEST.json`
- Modify: `ecup_matching/experiments/CURRENT.json`
- Modify: `docs/agent-memory/PROJECT_STATE.md`
- Modify: `docs/agent-memory/EXPERIMENT_INDEX.md`
- Modify: `docs/agent-memory/DECISIONS.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: approved v15 design and historical anchors v7/v12/v13.
- Produces: canonical source-of-truth contract all later jobs reference.

- [ ] **Step 1: Add v15 experiment files with immutable split/runtime/label-policy invariants and empty retained-results state.**
- [ ] **Step 2: Update CURRENT so v15 is primary `in_progress`, while explicitly recording v14/A17 as an untouched parallel completion track.**
- [ ] **Step 3: Update PROJECT_STATE, EXPERIMENT_INDEX and DECISIONS with v13 Public `0.3783781653`, v12 best `0.3798116204`, v14 negative item-centric evidence, and the v15 pivot.**
- [ ] **Step 4: Update AGENTS mandatory reading and immediate next action to the v15 contract.**
- [ ] **Step 5: Run memory policy/ingest/checkpoint only after repository tests are green; do not claim Memora checkpoint before verification.**

### Task 2: Deterministic field parser — TDD

**Files:**
- Create: `ecup_matching/v15_fields.py`
- Test: `ecup_matching/tests/test_v15_fields.py`

**Interfaces:**
- Produces: `NormalizedItemFields` dataclass and `normalize_item_fields(name: str, attributes: str, category: str) -> NormalizedItemFields`.

- [ ] **Step 1: Write failing tests for valid JSON dicts, malformed attributes, nulls, deterministic key ordering, model/SKU extraction, Unicode normalization and conservative numeric/unit normalization.**
- [ ] **Step 2: Run `pytest ecup_matching/tests/test_v15_fields.py -q` and require RED for missing implementation.**
- [ ] **Step 3: Implement pure deterministic parsing with no network/global learned state and no exceptions on malformed input.**
- [ ] **Step 4: Re-run the test file and require GREEN.**

### Task 3: Symmetric typed pair evidence — TDD

**Files:**
- Create: `ecup_matching/v15_pair_features.py`
- Test: `ecup_matching/tests/test_v15_pair_features.py`

**Interfaces:**
- Consumes: `NormalizedItemFields`.
- Produces: `build_pair_features(a, b) -> np.ndarray` and `PAIR_FEATURE_NAMES` with fixed ordering.

- [ ] **Step 1: Write failing tests asserting `features(a,b) == features(b,a)` and expected behavior for same/conflicting model, SKU, capacity, numeric tokens, brand and attribute keys.**
- [ ] **Step 2: Run targeted pytest and require RED.**
- [ ] **Step 3: Implement only deterministic commutative features with a frozen feature-name order.**
- [ ] **Step 4: Re-run tests and require GREEN.**

### Task 4: Field-aware serialization — TDD

**Files:**
- Create: `ecup_matching/v15_serialization.py`
- Test: `ecup_matching/tests/test_v15_serialization.py`

**Interfaces:**
- Consumes: normalized fields.
- Produces: `serialize_item(fields, side: str) -> str` and `serialize_pair(a,b) -> tuple[str,str]` or a frozen single-pair representation compatible with the selected tokenizer API.

- [ ] **Step 1: Write tests for stable field markers, deterministic attribute order, truncation pre-budget policy and absence of raw malformed JSON leakage.**
- [ ] **Step 2: Require RED.**
- [ ] **Step 3: Implement serialization without model-specific side effects.**
- [ ] **Step 4: Require GREEN.**

### Task 5: Single-model v15 neural head — TDD

**Files:**
- Create: `ecup_matching/v15_model.py`
- Test: `ecup_matching/tests/test_v15_model.py`

**Interfaces:**
- Produces: `V15Matcher(backbone, typed_feature_dim, num_categories, use_typed_features, use_category_head)` with one scalar logit per pair.

- [ ] **Step 1: Write CPU-unit tests with a tiny fake backbone verifying output shape, optional feature fusion, optional category residual, deterministic state dict keys and absence of second-backbone parameters.**
- [ ] **Step 2: Require RED.**
- [ ] **Step 3: Implement the minimal shared pooled representation + typed-feature MLP + category residual architecture.**
- [ ] **Step 4: Require GREEN.**

### Task 6: Canonical validation v5 — TDD

**Files:**
- Create: `ecup_matching/v15_validate.py`
- Test: `ecup_matching/tests/test_v15_validation.py`

**Interfaces:**
- Produces: `compute_macro_ap(...)`, per-category diagnostics, OOF integrity checker and hard-slice report.

- [ ] **Step 1: Write tests proving sklearn `average_precision_score` semantics, 20-category unweighted macro averaging, duplicate-row rejection, missing-row rejection and zero-overlap contract checks.**
- [ ] **Step 2: Require RED.**
- [ ] **Step 3: Implement validation helpers with fail-closed invariant checks.**
- [ ] **Step 4: Require GREEN.**

### Task 7: A0-A4 training entry point

**Files:**
- Create: `ecup_matching/v15_train.py`
- Test: extend `test_v15_model.py` / `test_v15_validation.py` with argument/config tests.

**Interfaces:**
- Consumes: canonical row map, human items/matches, cached ruBERT snapshot.
- Produces: checkpoint, OOF parquet, metrics JSON containing architecture variant, split/rowmap SHAs, label sources, fold, AP and runtime.

- [ ] **Step 1: Add failing CLI/config tests for variants `a0_field`, `a1_attrs`, `a2_typed`, `a3_category`, `a4_macro` and human-only label policy.**
- [ ] **Step 2: Require RED.**
- [ ] **Step 3: Implement one shared trainer where variants toggle one causal change at a time.**
- [ ] **Step 4: Compile/import smoke + targeted tests GREEN.**

### Task 8: Fold-safe active distillation contracts — TDD

**Files:**
- Create: `ecup_matching/v15_distill.py`
- Test: `ecup_matching/tests/test_v15_distill.py`

**Interfaces:**
- Produces: unlabelled candidate selector that requests only `id1,id2`; teacher score schema; held/sealed item exclusion; active-hardness sampler.

- [ ] **Step 1: Write a test double parquet reader that fails if `target` is requested and prove B1 requests only `id1,id2`.**
- [ ] **Step 2: Write tests excluding every human-universe item identity from fold-safe distill pools when required by the frozen policy.**
- [ ] **Step 3: Write deterministic active-sampling tests for model/SKU/numeric conflict and disagreement priority.**
- [ ] **Step 4: Require RED, implement minimal contracts, then require GREEN.**

### Task 9: Runtime path — TDD

**Files:**
- Create: `ecup_matching/v15_runtime.py`
- Test: `ecup_matching/tests/test_v15_runtime.py`

**Interfaces:**
- Produces: referenced-item-only loader, normalized-field cache, batched pair inference, exact ordered CSV writer.

- [ ] **Step 1: Write tests proving no unrelated item rows are materialized, repeated item parsing is cached once, input pair order is preserved, predictions are finite continuous numerics, output columns are exactly `id1,id2,predict`.**
- [ ] **Step 2: Require RED.**
- [ ] **Step 3: Implement minimal runtime path with no training-time dependency.**
- [ ] **Step 4: Require GREEN.**

### Task 10: Private immutable job contract

**Files:**
- Create in `gpu-dispatch`: `v15_job_contract.py`
- Test: `tests/test_v15_job_contract.py`
- Create: `docs/v15-executor-contract.md`

**Interfaces:**
- Produces: `validate_job_manifest(dict) -> None`, refusing unknown architecture family, missing 40-char source SHA, wrong split/rowmap SHA, legacy LLM label policy in A-family jobs, or production role without promotion evidence.

- [ ] **Step 1: Write failing manifest tests.**
- [ ] **Step 2: Require RED in trusted/unit environment.**
- [ ] **Step 3: Implement fail-closed validator.**
- [ ] **Step 4: Require GREEN.**

### Task 11: Queue A0/A1/A2 causal screens without touching v14

**Files:**
- Create in `gpu-dispatch`: `v15_a0_job.json`, `v15_a1_job.json`, `v15_a2_job.json`.
- Create workflows: `.github/workflows/ecup-v15-a0.yml`, `ecup-v15-a1.yml`, `ecup-v15-a2.yml`.

**Interfaces:**
- Each job binds exact public source SHA and one architecture variant.
- A1 and A2 workflows may queue behind the runner but must not cancel or mutate A17 or other v14 runs.

- [ ] **Step 1: Freeze manifests with fold0, human-only labels, exact public source/split/rowmap SHAs and one causal variant per job.**
- [ ] **Step 2: Add workflow provenance assertions before training.**
- [ ] **Step 3: Run compile/unit gate before GPU training.**
- [ ] **Step 4: Train on canonical fold0, upload only checkpoint/OOF/metrics/log evidence needed for analysis.**
- [ ] **Step 5: Add result gate that records but does not fabricate promotion when delta is below threshold.**

### Task 12: Queue bounded A3/A4 follow-ups and teacher preparation

**Files:**
- Add manifests/workflows only after A0-A2 code path is validated.

**Interfaces:**
- A3/A4 are independent screens using the same public source SHA family.
- B0 teacher workflow is allowed to prepare/train but B1/B2 pseudo-label generation must fail closed unless teacher-quality gate is satisfied.

- [ ] **Step 1: Queue A3 category-head control and A4 macro-oriented control as separate jobs.**
- [ ] **Step 2: Add B0 teacher screen with a predeclared teacher-vs-student held-human improvement gate.**
- [ ] **Step 3: Do not queue B2 distillation if B0 does not materially beat the selected student.**

### Task 13: Strict five-fold and production gates

**Files:**
- Create: `.github/workflows/ecup-v15-strict.yml`
- Create: `.github/workflows/ecup-v15-production.yml`
- Create/modify public `v15_build_submission.py`.

**Interfaces:**
- Strict workflow accepts only an explicitly promoted immutable config.
- Production workflow accepts only strict OOF evidence and outputs exact ZIP provenance/runtime evidence.

- [ ] **Step 1: Implement strict workflow gate requiring recorded strong fold0 candidate and immutable manifest.**
- [ ] **Step 2: Execute folds 1-4 with `max-parallel: 1` on the single self-hosted GPU and aggregate exactly 285210 OOF rows.**
- [ ] **Step 3: Implement production gate requiring five-fold evidence; refit full dev, build one-model ZIP and perform exact-byte organizer-shaped runtime tests.**
- [ ] **Step 4: Only a passing production artifact may be published to private HF with SHA roundtrip verification.**

### Task 14: Verification and morning handoff

**Files:**
- Update canonical RESULTS/SAFE_METRICS/index/state only for completed evidence.

- [ ] **Step 1: Run relevant public pytest suite and private executor tests.**
- [ ] **Step 2: Run `memory_policy.py`, `memory_ingest.py`, and checkpoint only from GREEN canonical repository state.**
- [ ] **Step 3: Record every completed overnight run with run ID, source SHA, fold Macro AP, 20-category evidence when available, runtime and decision.**
- [ ] **Step 4: Morning handoff clearly separates: ready-to-package keepers, promising screens needing strict OOF, rejected experiments, running jobs, and v14/A17 results that finished independently.**
- [ ] **Step 5: Never call a ZIP upload-ready unless exact final-byte Check/Public/Private-shaped runtime and CSV-format gates pass.**

## Self-review

- Spec coverage: architecture, label quarantine, field parser, typed fusion, category specialization, macro metric, active distillation, runtime, repo roles, immutable manifests, strict OOF and packaging are each assigned to a concrete task.
- Placeholder scan: no implementation task relies on a TODO/TBD placeholder; optional C relabeling is explicitly outside the overnight critical path.
- Type/interface consistency: parser -> pair features/serialization -> model/trainer -> validation/runtime; distillation is train-time only; private manifests point to exact public source.
- Scope boundary: overnight work prioritizes A-family causal screens plus B0 teacher preparation. Strict OOF/production exists as gated infrastructure and must not consume GPU merely because a job exists.
