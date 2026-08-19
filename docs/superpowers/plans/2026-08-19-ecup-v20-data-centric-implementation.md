# E-CUP v20 Audited Data-Centric Rationale Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and queue an end-to-end v20 data-centric product-matching pipeline that audits weak supervision, generates fold-safe real-item candidate pairs, statistically admits two-teacher rationale labels, trains data-only and multi-task RuBERT ablations, validates them on human/weak/proxy axes, and packages a one-checkpoint submission only after preregistered gates pass.

**Architecture:** Reuse the frozen v19/v18 split, weak-holdout and RuBERT runtime utilities. Add small focused v20 modules for semantic strata, Wilson admission, pair generation, teacher-schema validation, source-aware sampling/loss, proxy selection and promotion. Heavy data/model stages run from a private `gpu-dispatch` DAG with `queue: max`; CPU/M1 stages run before GPU and persist private bronze/silver/gold outputs on the runner.

**Tech Stack:** Python 3.12, pandas, NumPy, PyArrow, scikit-learn, PyTorch/Transformers, GitHub Actions, Docker, existing E-CUP organizer image and trusted GPU image.

**Spec:** `docs/superpowers/specs/2026-08-19-ecup-v20-audited-rationale-distillation-design.md`

## Global Constraints

- Frozen human split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed gold remains unopened and unscored for model selection.
- No generated/audited/training pair may contain an endpoint from the held human fold.
- Historical weak target is pseudo-supervision, never authoritative truth.
- New LLM-generated labels require two independent teacher lines plus deterministic checker compatibility; otherwise they are excluded.
- `UNCERTAIN` and teacher disagreement always receive training weight zero.
- Wilson lower-bound floors: positive `0.985`, negative `0.995`, category aggregate `0.970`, critical conflict `0.950`.
- Production runtime remains one pinned `ai-forever/ruBert-base` pair CrossEncoder, max length `256`, one `.safetensors`, no external service.
- v19 refresh is inherited only when v19 gate evidence says it passed.
- Concurrency group is `ecup-isolated-gpu` with `queue: max`; v20 is queued behind existing v18/v19 work.

---

### Task 1: v20 experiment ledger and policy contract

**Files:**
- Create: `ecup_matching/experiments/v20/PLAN.md`
- Create: `ecup_matching/experiments/v20/LEDGER.json`
- Create: `ecup_matching/ml/v20_policy.py`
- Test: `tests/test_v20_policy.py`

**Interfaces:**
- Produces `V20Policy`, `policy_sha256(policy)`, `validate_fold_exclusion(frame, forbidden_ids)`, and immutable v7/v12/v13B/v14 Public-LB anchors.

- [ ] **Step 1: Write failing policy tests.** Verify exact admission floors, one-checkpoint runtime contract, canonical anchors, and that any forbidden endpoint raises.
- [ ] **Step 2: Run `pytest -q tests/test_v20_policy.py` and observe failure because `v20_policy` does not exist.**
- [ ] **Step 3: Implement `v20_policy.py` with a frozen dataclass and deterministic JSON SHA-256.**
- [ ] **Step 4: Re-run the test and require PASS.**
- [ ] **Step 5: Add v1-v19 ledger entries for retained/rejected mechanisms and commit.**

### Task 2: semantic strata and full-corpus audit

**Files:**
- Create: `ecup_matching/ml/v20_strata.py`
- Create: `ecup_matching/ml/run_v20_data_audit.py`
- Test: `tests/test_v20_strata.py`

**Interfaces:**
- Produces `classify_pair_stratum(left, right) -> PairStratum`, `difficulty_bin(...)`, `audit_pair_frame(...)` and `STRATA.json`.

- [ ] **Step 1: Write failing tests for model/capacity/size/accessory/brand/sparse-evidence reason classification, symmetry and deterministic difficulty bins.**
- [ ] **Step 2: Verify RED with `pytest -q tests/test_v20_strata.py`.**
- [ ] **Step 3: Implement deterministic extractors from item name/category/attribute JSON; do not use target to select a reason.**
- [ ] **Step 4: Implement a PyArrow streaming audit that joins only referenced IDs in bounded batches and writes counts by `category × reason × difficulty × target_band`.**
- [ ] **Step 5: Run tests GREEN and commit.**

### Task 3: human calibration split and Wilson admission

**Files:**
- Create: `ecup_matching/ml/v20_admission.py`
- Test: `tests/test_v20_admission.py`

**Interfaces:**
- Produces `wilson_lower_bound(successes, trials, z=1.959963984540054)`, `build_fold_safe_audit_split(...)`, `admit_strata(audit_rows, policy)`.

- [ ] **Step 1: Write failing tests for Wilson boundaries, insufficient support rejection, positive/negative floors, critical-stratum floor and item-disjoint calibration split.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement admission with no fallback pooling and exact boundary semantics (`LCB >= floor`).**
- [ ] **Step 4: Verify GREEN and commit.**

### Task 4: target-free real-item candidate generation

**Files:**
- Create: `ecup_matching/ml/v20_candidates.py`
- Create: `ecup_matching/ml/run_v20_generate_candidates.py`
- Test: `tests/test_v20_candidates.py`

**Interfaces:**
- Produces deterministic candidate rows with `id1,id2,category,stratum,reason_code,generator_version,fold_exclusion_sha256`; no target column is allowed in generator inputs.

- [ ] **Step 1: Write failing tests proving candidate generation is target-free, symmetric-canonical, held-fold safe, duplicate-collapsing and degree/reason capped.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement same-category blocking using normalized model codes, brand, title-token signatures, numeric/critical-attribute signatures and accessory cues.**
- [ ] **Step 4: Add deterministic per-category/reason/anchor/item caps and manifest statistics.**
- [ ] **Step 5: Verify GREEN and commit.**

### Task 5: two-teacher rationale schema and label admission

**Files:**
- Create: `ecup_matching/ml/v20_teacher.py`
- Create: `ecup_matching/ml/run_v20_teacher_audit.py`
- Create: `ecup_matching/ml/run_v20_admit_labels.py`
- Test: `tests/test_v20_teacher.py`

**Interfaces:**
- `TeacherDecision` validates `MATCH|NON_MATCH|UNCERTAIN` plus the frozen reason-code vocabulary.
- `consensus_label(a,b,checker)` returns an admitted hard/auxiliary label only when two teachers agree and checker compatibility passes.

- [ ] **Step 1: Write failing tests for malformed JSON, single-teacher rejection, disagreement rejection, UNCERTAIN rejection, deterministic-checker conflict rejection, consensus acceptance and provenance hashes.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement strict schema parsing and two-teacher consensus.**
- [ ] **Step 4: Implement human audit report and statistical admission by Task 3 policy.**
- [ ] **Step 5: Verify GREEN and commit.**

### Task 6: source-aware gold corpus builder

**Files:**
- Create: `ecup_matching/ml/v20_corpus.py`
- Create: `ecup_matching/ml/run_v20_build_corpus.py`
- Test: `tests/test_v20_corpus.py`

**Interfaces:**
- Produces fold-safe gold rows with `source`, `target`, `match_weight`, optional auxiliary reason targets/masks and deterministic sampling keys.

- [ ] **Step 1: Write failing tests for human weight `1.0`, weak reliability cap `<1`, admitted-generated cap `<1`, uncertainty weight `0`, class/category/reason balancing and held-fold exclusion.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement reliability-weight calculation and source-aware deterministic sampler.**
- [ ] **Step 4: Verify GREEN and commit.**

### Task 7: rationale multi-task RuBERT training and curriculum

**Files:**
- Create: `ecup_matching/ml/v20_neural.py`
- Create: `ecup_matching/ml/run_v20_probe.py`
- Create: `ecup_matching/ml/run_v20_production.py`
- Test: `tests/test_v20_neural.py`

**Interfaces:**
- `V20MultiTaskHead` exposes main match logit plus model/numeric/variant/accessory/reason logits during training.
- `compute_v20_loss(...)` masks missing auxiliary labels.
- Probe supports `data_only`, `rationale`, `mixed_replay`, `scaled` with identical runtime backbone.

- [ ] **Step 1: Write failing tests for masked auxiliary loss, source weights, symmetry consistency, data-only equivalence to single-head loss and production stripping of auxiliary heads.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement the multi-task wrapper around the existing RuBERT encoder, with `lambda_reason=0.15` and `lambda_consistency=0.05` frozen before real metrics.**
- [ ] **Step 4: Implement Phase A/B/C curriculum with fixed replay: B `human:other=1:2` by batch exposure; C `human:other=4:1`, LR multiplier `0.35`; optional v19 refresh only from explicit evidence flag.**
- [ ] **Step 5: Verify GREEN and commit.**

### Task 8: proxy calibration and v20 promotion gates

**Files:**
- Create: `ecup_matching/ml/v20_proxy.py`
- Create: `ecup_matching/ml/v20_promotion.py`
- Test: `tests/test_v20_proxy.py`
- Test: `tests/test_v20_promotion.py`

**Interfaces:**
- Proxy accepts aggregate metric vectors for v7/v12/v13B/v14 and marks an axis promotable only if its ordering agrees with `v14 > v12 > v13B > v7` under deterministic tie tolerance.
- Promotion requires a promotable external-anchor proxy, human safety and weak/audited-stratum improvement.

- [ ] **Step 1: Write failing tests rejecting human-fold-like misranking and accepting only correctly ordered proxy dimensions.**
- [ ] **Step 2: Write failing promotion boundary tests: proxy gain `>0.005`, human delta `>=-0.003`, audited-tail delta `>=-0.02`, no category worse than `-0.04`; scaled two-fold mean human delta `>=0`.**
- [ ] **Step 3: Verify RED.**
- [ ] **Step 4: Implement proxy/promotion modules with `math.isclose(..., abs_tol=1e-12)` threshold semantics.**
- [ ] **Step 5: Verify GREEN and commit.**

### Task 9: hosted smoke and private v20 executor DAG

**Files:**
- Create: `.github/workflows/ecup-v20-m1-smoke.yml`
- Create: `v20_m1_smoke_job.json`
- Private repo create: `v20_executor.py`
- Private repo create: `v20_job.json`
- Private repo create: `.github/workflows/ecup-v20-data-centric.yml`

**Interfaces:**
- Hosted smoke runs `py_compile` + all `tests/test_v20_*.py` and records MPS availability.
- Private DAG stages D0-D10 with `concurrency.group=ecup-isolated-gpu` and `queue:max`.

- [ ] **Step 1: Add hosted smoke config only after Task 1-8 tests exist.**
- [ ] **Step 2: Implement private `verify` stage that clones one exact v20 source SHA and runs py_compile/pytest before data/GPU work.**
- [ ] **Step 3: Add CPU D0/D1/D2, teacher D3/D4, GPU D5/D6/D7, conditional D8/D9/D10.**
- [ ] **Step 4: Ensure every stage writes an immutable decision JSON to persistent `/srv/github-gpu/output/v20-*`; artifact upload is best-effort only.**
- [ ] **Step 5: Commit and trigger by changing only `v20_job.json`.**

### Task 10: production finalizer and verification

**Files:**
- Private repo create: `v20_build_final.py`

**Interfaces:**
- Refuses packaging without D9 two-fold promotion, gold unopened, split SHA match, exactly one production `.safetensors`, exact source/policy hashes and one-head production metadata.

- [ ] **Step 1: Add finalizer contract assertions before invoking the proven v7-compatible package builder.**
- [ ] **Step 2: Run exact 1,000-pair organizer-shaped Check: network off, GPU enabled, exact schema/order, finite predictions, >10 unique scores, 60-second limit.**
- [ ] **Step 3: Audit ZIP for unsafe paths/symlinks, size <5 GiB, exact one checkpoint and required metadata.**
- [ ] **Step 4: Write `manifest.json`, `SHA256SUMS.txt`, promotion metrics, runtime and source/provenance hashes.**
- [ ] **Step 5: Persist exact ZIP on runner and upload split artifact best-effort.**

## Self-review

- Spec coverage: D0-D10, semantic audit, two-teacher admission, candidate generation, multi-task rationale, mixed replay, proxy calibration, two-fold confirmation, one-checkpoint packaging and sealed-gold constraints each have an explicit task.
- Placeholder scan: no TODO/TBD/implement-later placeholders are part of executable task requirements.
- Type consistency: policy/admission/corpus/proxy interfaces are explicitly named and consumed by later tasks.
- Causal isolation: D5 data-only, D6 rationale, D7 replay, D8 scale; no stage introduces two unmeasured mechanisms simultaneously.
