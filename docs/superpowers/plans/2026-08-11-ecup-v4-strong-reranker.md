# E-CUP v4 Strong Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, select and package a v4 E-CUP matcher that beats retained v3 Macro AP `0.5254642645846543` on the unchanged item-disjoint validation split.

**Architecture:** Preserve v2b structured scoring as the stable anchor, replace the tiny neural branch with `ai-forever/ruBert-base`, then measure three immutable curriculum stages: v4a full human, v4b high-confidence weak-label continuation, and v4c model-mined hard negatives with 50% ordinary replay. Train on the isolated home RTX 2060 SUPER through the private dispatcher; only after a quality winner exists build an organizer-compatible offline package.

**Tech Stack:** Python 3.11, pandas/pyarrow/numpy/scikit-learn, PyTorch CUDA AMP, Hugging Face Transformers, GitHub Actions self-hosted runner, Docker, private Hugging Face dataset storage.

## Global Constraints

- Work only on `ecup-matching-2026`; never modify or merge `main`.
- Keep canonical v3 artifact SHA-256 `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660` immutable.
- Fixed validation is exactly 73,131 human pairs with zero train/validation item overlap.
- Primary retention metric is unweighted category Macro Average Precision.
- Final v4 must strictly exceed `0.5254642645846543` or remain rejected while v3 stays current best.
- Raw data, model weights and submission ZIPs remain private.
- Base model is `ai-forever/ruBert-base`, Apache-2.0, pinned to an exact revision before retained training.
- GPU execution remains network-disabled for public-source code; model assets are baked into a trusted private dispatcher image.
- Intermediate local CPU/M1 inference time is not a rejection criterion; final organizer H100 compatibility is.

---

### Task 1: Declare v4 and repair canonical handoff state

**Files:**
- Create: `ecup_matching/experiments/v4/PLAN.md`
- Create: `ecup_matching/experiments/v4/RESULTS.md`
- Modify: `ecup_matching/experiments/CURRENT.json`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: retained v3 metrics and immutable artifact path.
- Produces: canonical v4 `in_progress` state for memory policy and future agents.

- [ ] **Step 1: Write v4 PLAN with exact baseline, v4a/v4b/v4c datasets, quality gates and immutable artifact layout.**
- [ ] **Step 2: Create RESULTS as an explicit in-progress experiment log, containing no fabricated metrics.**
- [ ] **Step 3: Change CURRENT to `version=v4`, `status=in_progress`, baseline v3 Macro AP `0.5254642645846543`.**
- [ ] **Step 4: Update AGENTS current state from stale v2/v3 wording to retained v3 + active v4.**
- [ ] **Step 5: Verify `python scripts/memory_policy.py` in GitHub Actions.**

### Task 2: Add v4 curriculum contracts TDD-first

**Files:**
- Create: `ecup_matching/tests/test_v4_curriculum.py`
- Create: `ecup_matching/ml/v4_curriculum.py`

**Interfaces:**
- Consumes: serialized v3/v2 reranker frames with `id1,id2,target,category,sample_weight,text_a,text_b,source`.
- Produces:
  - `assert_item_disjoint(train, valid) -> None`
  - `build_human_curriculum(frame) -> pd.DataFrame`
  - `build_weak_curriculum(human, weak, max_weak_rows, seed=2026) -> pd.DataFrame`
  - `build_hard_replay_curriculum(parent, mined_negatives, positives, hard_fraction=0.25, positive_fraction=0.25, seed=2026) -> pd.DataFrame`

- [ ] **Step 1: RED — test full-human curriculum never compacts rows and preserves every authoritative row.**

```python
def test_full_human_curriculum_keeps_all_authoritative_rows():
    out = build_human_curriculum(frame)
    assert len(out) == len(frame)
    assert set(out.source) == {"human"}
```

- [ ] **Step 2: RED — test weak curriculum rejects validation-item leakage and caps weak rows deterministically.**

```python
def test_weak_curriculum_is_disjoint_and_capped():
    assert_item_disjoint(out, valid)
    assert (out.source == "weak").sum() == 6
```

- [ ] **Step 3: RED — test hard replay composition is 25% hard negatives, 25% positives and 50% ordinary replay for a divisible fixture.**
- [ ] **Step 4: Run `python -m pytest ecup_matching/tests/test_v4_curriculum.py -q`; expected failure because `v4_curriculum` does not exist.**
- [ ] **Step 5: GREEN — implement only the tested deterministic curriculum helpers.**
- [ ] **Step 6: Run new tests plus the full `ecup_matching/tests` suite.**

### Task 3: Add the v4 CUDA trainer TDD-first

**Files:**
- Create: `ecup_matching/tests/test_train_v4_reranker.py`
- Create: `ecup_matching/ml/train_v4_reranker.py`

**Interfaces:**
- Consumes raw competition parquet paths, v2 structured validation predictions, and a local pinned base-model directory.
- Produces `metrics.json`, `manifest.json`, `validation_predictions.parquet`, `model/`, and immutable stage directories `v4a-model/`, `v4b-model/`, `v4c-model/` inside one run output.

- [ ] **Step 1: RED — test model selection never accepts a candidate at or below v3 and returns the best strictly higher Macro AP.**

```python
def test_select_v4_candidate_requires_strict_v3_improvement():
    result = select_v4_candidate({"v4a": 0.525, "v4b": 0.541, "v4c": 0.539})
    assert result == ("v4b", 0.541)
```

- [ ] **Step 2: RED — test category-alpha shrinkage keeps every category alpha between global alpha and the raw category optimum according to configured shrinkage.**
- [ ] **Step 3: RED — test v4 metrics payload records 73,131 validation rows, zero overlap, all 20 per-category APs, base model, revision, CUDA device and all three candidate outcomes.**
- [ ] **Step 4: Run focused tests and confirm missing-module failure.**
- [ ] **Step 5: GREEN — implement trainer using existing `prepare_training_examples`, `_train_model`, `_evaluate`, `_predict`, `select_best_blend`, v4 curriculum helpers and exact item-disjoint guards.**
- [ ] **Step 6: v4a uses all human rows, FP16/BF16 CUDA, micro-batch default 4, gradient accumulation 8, max length 256.**
- [ ] **Step 7: v4b warm-starts from v4a, uses up to 600k high-confidence weak rows with human-dominant weighting and lower LR.**
- [ ] **Step 8: v4c mines difficult negatives from the parent model and fine-tunes on the 25/25/50 replay curriculum at lower LR.**
- [ ] **Step 9: Evaluate neural-only, global v2b blend and shrinkage category blend after every stage; save every stage before later mutation.**
- [ ] **Step 10: Run full repository tests and memory policy.**

### Task 4: Extend the private dispatcher safely for v4

**Files in private `MakSoS1/gpu-dispatch`:**
- Modify: `tests/test_dispatch_contract.py`
- Modify: `tests/test_run_job.py`
- Modify: `tests/test_image_policy.py`
- Modify: `dispatch_contract.py`
- Modify: `run_job.py`
- Modify: `Dockerfile`
- Modify: `.github/workflows/ecup-gpu.yml`

**Interfaces:**
- Adds trusted profiles `v4-benchmark` and `v4-train`.
- Uses trusted image `ecup-gpu-trusted:2026-08-11-v4` containing pinned `rubert-tiny2` plus pinned `ai-forever/ruBert-base`.
- Public source still receives no token, network, Docker socket, Windows mount or writable host source/data path.

- [ ] **Step 1: RED — dispatcher contract rejects unknown profiles but accepts `v4-benchmark` and `v4-train`.**
- [ ] **Step 2: RED — run-job tests assert the v4 command invokes `python -m ecup_matching.ml.train_v4_reranker`, local `/opt/models/rubert-base`, and fixed safe arguments.**
- [ ] **Step 3: RED — image policy test requires exact pinned model revision and read-only model directory.**
- [ ] **Step 4: Run private dispatcher pytest and confirm RED.**
- [ ] **Step 5: GREEN — extend profiles without adding arbitrary user-provided command arguments.**
- [ ] **Step 6: Bake `ai-forever/ruBert-base` exact revision into trusted image and keep public execution offline.**
- [ ] **Step 7: `v4-benchmark` runs a short fixed CUDA workload; `v4-train` runs the complete immutable v4 ladder.**
- [ ] **Step 8: Run all private dispatcher tests; require zero failures.**

### Task 5: Execute measured RTX benchmark and production v4 training

**Files:** no public source changes except evidence documentation after the run.

**Interfaces:**
- Input: exact 40-char SHA reachable from `ecup-matching-2026`.
- Output: `/srv/github-gpu/output/<run-id>/` with model/metrics/derived data, validated and copied out of the isolated container.

- [ ] **Step 1: Dispatch `v4-benchmark` on the exact tested public SHA.**
- [ ] **Step 2: Record GPU name, CUDA version, peak allocated VRAM, examples/sec and effective batch configuration.**
- [ ] **Step 3: If CUDA OOM occurs, apply the predetermined fallback in order: micro-batch 4→2→1, enable gradient checkpointing, max length 256→192. Do not remove human rows.**
- [ ] **Step 4: Dispatch `v4-train`.**
- [ ] **Step 5: Validate returned metrics before treating the run as evidence: 73,131 validation rows, overlap 0, 20 AP categories, finite scores, NVIDIA CUDA device.**
- [ ] **Step 6: Compare v4a, v4b and v4c against retained v3 and select only the best strict improvement.**

### Task 6: Add organizer submission v4 TDD-first

**Files:**
- Create: `ecup_matching/tests/test_submission_v4.py`
- Create: `ecup_matching/build_submission_v4.py`
- Reuse/modify only as required: `ecup_matching/submission/run_v3.py` or create `ecup_matching/submission/run_v4.py`

**Interfaces:**
- Consumes selected v4 model directory, manifest, v2 structured model assets.
- Produces organizer ZIP containing `metadata.json`, runtime source, structured assets, tokenizer/model assets and no credentials/raw training data.

- [ ] **Step 1: RED — package test requires `metadata.json`, complete model/tokenizer files, v4 manifest and no secret/raw-data filenames.**
- [ ] **Step 2: RED — runtime contract test proves CLI accepts organizer `--items_path`, `--matches_path`, `--output_path` aliases and preserves input pair order.**
- [ ] **Step 3: GREEN — implement v4 package by adapting verified v3 runtime, retaining continuous scores and CUDA auto-selection.**
- [ ] **Step 4: Run all public tests.**

### Task 7: Exact organizer-image verification and canonical freeze

**Files:**
- Create/modify GitHub workflow only if the existing v3 final-package harness cannot parameterize v4 safely.

- [ ] **Step 1: Build ZIP inside `odsai/ecup26-matching-baseline:1.0`.**
- [ ] **Step 2: Run `--network none` correctness smoke and verify row count/order/schema/finite score range.**
- [ ] **Step 3: Exercise CUDA inference on the home RTX with the same packaged model/runtime; record actual neural pair count and throughput.**
- [ ] **Step 4: Check archive <5 GB and expected H100 feasibility; only add gating if all-pair inference is not safely feasible.**
- [ ] **Step 5: Compute ZIP SHA-256 and upload to `submissions/v4/canonical/<sha256>/...`; verify remote presence and checksum.**
- [ ] **Step 6: Create/update convenience alias only after canonical upload verification.**

### Task 8: Close v4 memory and handoff

**Files:**
- Modify: `ecup_matching/experiments/v4/RESULTS.md`
- Modify: `ecup_matching/experiments/CURRENT.json`
- Modify: `docs/agent-memory/EXPERIMENT_INDEX.md`
- Modify: `docs/agent-memory/PROJECT_STATE.md`
- Modify: `docs/agent-memory/DECISIONS.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Record exact source SHA, runner IDs, model revision, row counts, v4a/v4b/v4c Macro AP, all category APs, training/runtime evidence, artifact paths and failures.**
- [ ] **Step 2: If score beats v3 and package gates pass, mark v4 completed/retained; otherwise mark v4 rejected and restore v3 as current best without deleting v4 evidence.**
- [ ] **Step 3: Run `python scripts/memory_policy.py`.**
- [ ] **Step 4: Run `python scripts/memory_ingest.py`.**
- [ ] **Step 5: Run `python scripts/memory_checkpoint.py --iteration v4` and verify private HF checkpoint.**
- [ ] **Step 6: Run final public and private dispatcher test suites and record counts.**
