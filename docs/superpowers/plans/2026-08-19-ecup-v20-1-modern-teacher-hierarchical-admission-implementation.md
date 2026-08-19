# E-CUP v20.1 Modern Teacher + Hierarchical Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish v20 with empirical modern-teacher selection, statistically powered hierarchical admission, staged generated-label inference, and the existing causal model/promotion ladder.

**Architecture:** Preserve the v20.0 data audit/candidate/proxy/runtime path. Insert a teacher-bakeoff layer before candidate labeling, replace fine-stratum admission with pooled hierarchical gates, then reuse the existing data-only/rationale/replay/scaled/production stages. Private executor stages remain resumable through persisted campaign artifacts.

**Tech Stack:** Python 3, pandas/pyarrow, scipy-free Wilson math, PyTorch/Transformers for small HF teachers, llama.cpp-compatible GGUF backend for quantized Gemma/Qwen where needed, Docker, GitHub Actions, RTX 2060 SUPER 8 GB.

**Spec:** `docs/superpowers/specs/2026-08-19-ecup-v20-1-modern-teacher-hierarchical-admission-design.md`

## Global Constraints
- Production runtime: one `ai-forever/ruBert-base` pair CrossEncoder, max length 256, one safetensors checkpoint.
- Split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Gold 80,444 rows never opened/scored.
- Human-item exclusion for generated training candidates; human+historical-weak exclusion for proxy.
- Teacher VRAM cap: 7.75 GiB.
- Wilson floors stay 0.985 MATCH, 0.995 NON_MATCH, 0.970 category, 0.950 critical.
- Final promotion still requires proxy > calibrated v14 plus the existing human/tail/category/two-fold gates.

---

### Task 1: Hierarchical admission contract

**Files:**
- Modify: `ecup_matching/ml/v20_admission.py`
- Modify: `ecup_matching/ml/v20_policy.py`
- Create: `tests/test_v20_hierarchical_admission.py`

**Interfaces:**
- Produces `build_hierarchical_policy(audit_rows, policy) -> dict`
- Produces `row_passes_hierarchical_policy(row, policy_report) -> bool`

- [ ] Write failing tests proving 100/100 cannot satisfy the 0.995 Wilson floor and that pooled reason/category support can satisfy it without lowering the floor.
- [ ] Add policy helpers that report predicted-label, reason, category, and critical-family Wilson records.
- [ ] Require every applicable gate for row admission.
- [ ] Preserve the existing fine-stratum function for backward evidence but stop using it in v20.1.
- [ ] Run `pytest -q tests/test_v20_admission.py tests/test_v20_hierarchical_admission.py`.

### Task 2: Teacher bakeoff scoring and pair selection

**Files:**
- Create: `ecup_matching/ml/v20_teacher_bakeoff.py`
- Create: `tests/test_v20_teacher_bakeoff.py`

**Interfaces:**
- `score_teacher(audit_truth, normalized_labels, runtime_manifest) -> dict`
- `score_pair(first_labels, second_labels, audit_truth) -> dict`
- `select_teacher_pair(teacher_reports, pair_reports) -> dict`

- [ ] Write failing tests for JSON-rate/coverage/LCB/VRAM hard eligibility.
- [ ] Write a test that same-family or same-revision pair selection is rejected.
- [ ] Implement deterministic ranking: consensus precision → critical precision → coverage → throughput.
- [ ] Fail closed when no eligible pair exists.

### Task 3: Deterministic stratified bakeoff slice

**Files:**
- Create: `ecup_matching/ml/run_v20_prepare_teacher_bakeoff.py`
- Create: `tests/test_v20_teacher_bakeoff_slice.py`

**Interfaces:**
- CLI consumes fold audit parquet and writes `teacher-bakeoff.parquet` + manifest.

- [ ] Test deterministic row selection with per-reason/category/label coverage and maximum 4,000 rows/fold.
- [ ] Implement target-blind prompt input columns while keeping truth in a separate audit truth artifact.
- [ ] Assert no held-fold/human split overlap change.

### Task 4: Backend-agnostic teacher runner

**Files:**
- Modify: `ecup_matching/ml/run_v20_teacher_label.py`
- Create: `ecup_matching/ml/v20_teacher_backend.py`
- Create: `tests/test_v20_teacher_backend.py`

**Interfaces:**
- Normalize `transformers`, `gguf` and `seq2seq` outputs into the existing `TeacherDecision` schema.
- Runtime manifest records exact model/revision/backend/quantization/rows_sec/peak_vram_gib.

- [ ] Test backend configuration validation and normalized provenance.
- [ ] Add `--backend`, `--quantization`, `--model-file` options without changing the JSON schema.
- [ ] Add deterministic non-sampling generation for audit labels.
- [ ] Record OOM/backend failure as machine-readable ineligible evidence instead of corrupt partial success.

### Task 5: Audit-only teacher bakeoff driver

**Files:**
- Create: `ecup_matching/ml/run_v20_teacher_bakeoff.py`
- Create: `tests/test_v20_teacher_bakeoff_driver.py`

**Interfaces:**
- Consumes audit truth + each teacher label file/manifest.
- Writes `teacher-bakeoff-report.json` and `selected-teachers.json`.

- [ ] Test that Yandex is not in canonical candidate configuration.
- [ ] Test candidates include Qwen3.5-4B, Gemma4-E2B-it, EuroLLM-1.7B-Instruct, FRED-T5-1.7B control.
- [ ] Implement per-fold and combined teacher/pair metrics.
- [ ] Require two-fold eligibility for selected pair.

### Task 6: Hierarchical generated-label admission

**Files:**
- Modify: `ecup_matching/ml/run_v20_admit_labels.py`
- Modify: `ecup_matching/ml/run_v20_intersect_labels.py`
- Create: `tests/test_v20_hierarchical_generated_admission.py`

**Interfaces:**
- `audit` mode writes hierarchical policy.
- `candidates` mode uses selected teacher pair + fold policy.

- [ ] Write tests for predicted-label/reason/category/critical conjunction.
- [ ] Ensure deterministic-checker conflicts and UNCERTAIN rows never pass.
- [ ] Intersect fold0/fold1 labels and keep minimum empirical reliability.

### Task 7: Staged candidate queue

**Files:**
- Modify: `ecup_matching/ml/run_v20_combine_teacher_pairs.py`
- Create: `ecup_matching/ml/run_v20_filter_candidate_queue.py`
- Create: `tests/test_v20_staged_teacher_queue.py`

**Interfaces:**
- Audit queue is built separately from candidate queue.
- Candidate queue is filtered by reason/category gates before expensive inference.

- [ ] Test audit rows are never mixed with generated candidates before pair selection.
- [ ] Test candidate queue only contains policy-eligible reason/category groups and remains target-free.
- [ ] Emit queue-size/estimated-token manifests.

### Task 8: Private executor v20.1 campaign

**Files in `MakSoS1/gpu-dispatch` branch `ecup-v20-executor`:**
- Modify: `v20_job.json`
- Modify: `v20_executor.py`
- Modify: `v20_dispatch.py`
- Modify: `.github/workflows/ecup-v20-sequential-fixed.yml`
- Add/modify executor tests.

**Interfaces:**
- New sequence: verify → prepare/reuse D0-D1 → fold bakeoff slices → run candidate teachers → select pair → build hierarchical policies → filtered candidate labeling → admit/intersect → anchors → control/data/rationale/replay → scaled folds → confirm → production → package.

- [ ] Pin public source exact SHA after Tasks 1–7.
- [ ] Add teacher candidate configuration with exact requested revisions resolved at runtime.
- [ ] Prefer quantized backend for Qwen3.5/Gemma4; retain full-precision EuroLLM/FRED controls.
- [ ] Reuse an existing compatible D0/D1 campaign root when source-independent artifact checksums/schema match; otherwise rebuild.
- [ ] Keep one GPU-heavy stage at a time.
- [ ] Publish terminal custom status and persist all manifests even when no teacher pair passes.

### Task 9: Model ladder and promotion verification

**Files:**
- Reuse `v20_select.py`, `v20_promotion.py`, `run_v20_probe.py`, `run_v20_production.py` unless tests reveal an incompatibility.
- Add regression tests only where v20.1 metadata changes interfaces.

- [ ] Verify control remains identical in mechanism to v20 historical control.
- [ ] Verify data-only adds only admitted v20.1 labels.
- [ ] Verify rationale/replay/scale causal isolation.
- [ ] Verify proxy calibration still reproduces `v14 > v12 > v13B > v7`.
- [ ] Require selected candidate proxy > v14 and existing delta gates.

### Task 10: Production and final ZIP

**Files in private executor:**
- Modify `v20_build_final.py` only if new provenance fields are required.

- [ ] Require selected teacher pair report, both fold hierarchical policies, non-empty intersection, two-fold scaled promote, and gold flags false.
- [ ] Run full development refit with the selected exact causal mode.
- [ ] Build one production safetensors checkpoint.
- [ ] Run organizer image exact ZIP check and runtime gate.
- [ ] Write `manifest.json`, `SHA256SUMS.txt`, teacher provenance and final archive SHA-256.
- [ ] Upload split artifact best-effort; persistent runner path remains authoritative.
