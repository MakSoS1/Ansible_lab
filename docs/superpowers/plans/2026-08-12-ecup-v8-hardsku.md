# E-CUP v8 HardSKU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a v8 submission that improves real leaderboard transfer over v7 while preserving the E-CUP runtime/package limits, using evidence from test-like distribution diagnostics, graph-context ablations, hard-negative training, and hybrid ranking.

**Architecture:** Keep v7 as an immutable baseline. Add a separate v8 validation layer that measures human item-disjoint OOF and an explicitly diagnostic LLM-candidate distribution. Test graph-context rescoring as a target-free postprocessor, then test hard-negative training and a stronger ranking-pretrained teacher/backbone. Select only components that improve confirmatory folds and runtime, then package the smallest winning cascade/hybrid.

**Tech Stack:** Python 3.11, pandas/numpy/pyarrow, scikit-learn, PyTorch/Transformers, GitHub Actions, private Hugging Face competition mirror, RTX 2060 SUPER dispatcher, organizer Docker image.

## Global Constraints

- Real leaderboard reference supplied by user: v7 score `0.3655833314`; observed leaders around `0.46–0.48`.
- Official metric: unweighted mean of per-category sklearn `average_precision_score` over exactly 20 categories.
- Human split stays immutable: SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`, 365,654 rows total, 285,210 development, 80,444 sealed gold, zero cross-split item overlap.
- Sealed gold remains unopened throughout model selection.
- LLM-label diagnostics must never be called gold/strict AP; soft-label and high-confidence pseudo-label metrics are diagnostic only.
- Test-like diagnostics must exclude all human item IDs when measuring transfer to unseen-item candidate structure where possible.
- Any graph postprocessor must be target-free at inference and operate only on pair IDs, categories and model scores.
- Final archive <= 5 GB and uses organizer image `odsai/ecup26-matching-baseline:1.0`.
- Runtime hard limits retained from competition design: Check 1 min, Public 6 min, Private 13 min. Final acceptance requires exact organizer-container runtime evidence, not an H100 extrapolation.
- No model/component is selected from fold0 alone. Fold0 may screen candidates; folds 1–4 are confirmatory and full OOF is required before final packaging.
- Every production behavior change follows RED -> GREEN TDD.

---

### Task 1: Distribution-shift and prevalence audit

**Files:**
- Create: `ecup_matching/ml/v8_distribution.py`
- Create: `ecup_matching/tests/test_v8_distribution.py`
- Create: `.github/workflows/ecup-v8-distribution-audit.yml`
- Create: `ecup_matching/experiments/v8/RESULTS.md`
- Create: `ecup_matching/experiments/v8/SAFE_METRICS.json`

**Interfaces:**
- Produces: `binary_prevalence(frame, target_col='target') -> float`, `target_distribution(frame) -> dict`, `candidate_graph_summary(frame) -> dict`, `category_distribution_report(matches, item_categories) -> dict`, `human_excluded_llm_mask(llm_pairs, human_ids) -> np.ndarray`.

- [ ] **Step 1: Write failing unit tests** for binary prevalence, soft-label summary, graph degree statistics, category attachment and exclusion of every human ID from test-like LLM rows.
- [ ] **Step 2: Run `python -m pytest ecup_matching/tests/test_v8_distribution.py -q` and verify RED** because `v8_distribution` does not exist.
- [ ] **Step 3: Implement only the pure analysis functions** without model code or target leakage.
- [ ] **Step 4: Re-run targeted tests and full existing v7 tests; require GREEN.**
- [ ] **Step 5: Run a GitHub-hosted audit against private `matches.parquet`, `matches_llm.parquet`, `items_human.parquet` and streamed `items.parquet` category columns.** Emit aggregate-only JSON: human positive prevalence per category, LLM target distribution per category, high-confidence pseudo-positive rate, human-vs-LLM degree distributions, human-item-excluded LLM row count, and candidate-density ratios.
- [ ] **Step 6: Record the conclusion as CONFIRMED / PARTIAL / REJECTED.** In particular, never infer true LLM/test prevalence merely from mean soft targets; report both mean and high-confidence pseudo-label proportions.

### Task 2: Graph-context scorer

**Files:**
- Create: `ecup_matching/ml/v8_graph.py`
- Create: `ecup_matching/tests/test_v8_graph.py`
- Create: `ecup_matching/ml/run_v8_graph_oof.py`

**Interfaces:**
- Produces: `graph_features(pairs, scores) -> pd.DataFrame` with endpoint degree, endpoint score rank/percentile, reciprocal-best flag, reciprocal-top-k flags, top1 margin, local score z/rank and ambiguity measures; `graph_rescore(scores, features, config) -> np.ndarray`.

- [ ] **Step 1: Write failing tests** proving reciprocal-best symmetry, degree/rank correctness, permutation equivariance, target independence, finite scores, and no cross-category graph mixing.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement vectorized numpy/pandas graph features and a monotone bounded rescore.** Do not use labels or item metadata.
- [ ] **Step 4: Verify GREEN and benchmark on 275k synthetic pairs; graph feature + rescore wall time must be negligible relative to neural inference (target <5 s on GitHub CPU; hard reject if unexpectedly expensive).**
- [ ] **Step 5: Evaluate on existing honest OOF predictions first (v5 strict OOF and any available v7 held-fold predictions), tuning only on designated development folds and reporting untouched confirmatory folds.** Test no-graph, reciprocal-only, rank/ambiguity-only, and combined variants.
- [ ] **Step 6: Keep graph processing only if confirmatory macro AP improves and no category suffers a material unexplained regression.**

### Task 3: Test-like LLM validation protocol

**Files:**
- Create: `ecup_matching/ml/v8_testlike.py`
- Create: `ecup_matching/tests/test_v8_testlike.py`
- Create: `ecup_matching/ml/run_v8_testlike.py`

**Interfaces:**
- Produces: deterministic human-item-excluded LLM slices; `pseudo_binary_labels(target, low=0.05, high=0.95)` that drops ambiguous rows; category-macro pseudo AP; pairwise ordering accuracy on soft targets; distribution-weighted human OOF diagnostic.

- [ ] **Step 1: RED tests** for deterministic slicing, complete human-ID exclusion, ambiguous target removal, per-category support checks, and explicit `diagnostic_only=True` metadata.
- [ ] **Step 2: Implement the protocol.**
- [ ] **Step 3: GREEN targeted/full tests.**
- [ ] **Step 4: Score the same fixed slice with retained v5/v6/v7 candidates where artifacts permit.** The purpose is model ordering/correlation, not claiming absolute test AP.
- [ ] **Step 5: Compare model ordering against known leaderboard evidence (`v7=0.3655833314` and any prior submitted score available in experiment memory).** Prefer diagnostics that rank known submissions in the same order as the board.

### Task 4: Structured hard-negative curriculum

**Files:**
- Create: `ecup_matching/ml/v8_hardneg.py`
- Create: `ecup_matching/tests/test_v8_hardneg.py`
- Create: `ecup_matching/ml/run_v8_hardneg_probe.py`

**Interfaces:**
- Produces per-training-row target-free hardness from text containment plus explicit SKU/model/numeric/unit conflicts; sampler draws a configurable mixture of hardest negatives and random negatives within category.

- [ ] **Step 1: RED tests** showing near-identical names with conflicting model/quantity/size rank harder than unrelated negatives, deterministic sampling, no held-row access, and category-local sampling.
- [ ] **Step 2: Implement structured hardness and sampler.**
- [ ] **Step 3: GREEN tests.**
- [ ] **Step 4: Run controlled GPU probes against the exact v7 identity-v2 baseline: same source data/model/epoch/LR/batches; only hard-negative sampling changes.** Screen on fold0, then immediately run at least two untouched confirmatory folds for any positive candidate.
- [ ] **Step 5: Keep only if mean confirmatory improvement is positive and hard-slice AP improves.**

### Task 5: Ranking-pretrained model ablation

**Files:**
- Create: `ecup_matching/ml/v8_backbones.py`
- Create: `ecup_matching/tests/test_v8_backbones.py`
- Create: `ecup_matching/ml/run_v8_backbone_probe.py`

**Interfaces:**
- Candidate A: retained `ai-forever/ruBert-base` identity-v2.
- Candidate B: `Alibaba-NLP/gte-multilingual-reranker-base` (ranking-pretrained multilingual, Russian-supported, Apache-2.0; exact revision must be pinned before use).
- Optional teacher-only candidate: `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0; do not promote to runtime scorer until measured).

- [ ] **Step 1: RED tests** require exact model revision, supported sequence-classification score extraction, offline save/load closure, and license metadata field.
- [ ] **Step 2: Implement adapter only for models that pass revision/license inspection.**
- [ ] **Step 3: GREEN tests.**
- [ ] **Step 4: Run bounded off-the-shelf inference probes on identical held rows before expensive fine-tuning.** Reject candidates clearly weaker than v7 on hard-slice ranking.
- [ ] **Step 5: Fine-tune only the strongest feasible candidate on the v8 hard-negative curriculum and measure throughput/VRAM.**
- [ ] **Step 6: Runtime reject any full-pair scorer whose measured throughput cannot plausibly fit the 13-minute private limit with CPU/tokenization overhead; such a model may remain teacher-only.**

### Task 6: Hybrid/ensemble selection

**Files:**
- Create: `ecup_matching/ml/v8_hybrid.py`
- Create: `ecup_matching/tests/test_v8_hybrid.py`
- Create: `ecup_matching/ml/run_v8_hybrid_oof.py`

**Interfaces:**
- Combines neural score, retained v5/v6 structured score, graph features/rescore and explicit structured conflicts using cross-fitted category-safe models or fixed rank blends.

- [ ] **Step 1: RED tests** prohibit fitting on held-fold targets, enforce row alignment, category-local percentile transforms, finite output and deterministic inference.
- [ ] **Step 2: Implement simplest viable fixed-rank blend first; only add a learned meta-ranker if cross-fitting proves an advantage.**
- [ ] **Step 3: GREEN tests.**
- [ ] **Step 4: Run 5-fold OOF for finalists and report overall + each category + hard slice + fold scores.** No single-fold promotion.
- [ ] **Step 5: Select the Pareto winner using quality and runtime jointly.**

### Task 7: Production cascade and runtime gate

**Files:**
- Create/modify only after winner is known: `ecup_matching/submission/predict_v8.py`, `ecup_matching/submission/run_v8.py`, `ecup_matching/build_submission_v8.py`, `ecup_matching/tests/test_v8_submission_contract.py`.

**Interfaces:**
- Exact organizer CLI: `--output_path`, `--items_path`, `--matches_path`; output columns exactly `id1,id2,predict` in input order.

- [ ] **Step 1: RED submission-contract tests** for CLI/schema/order/offline imports/model closure/no credentials/no competition data/<=5GB metadata.
- [ ] **Step 2: Implement production refit and inference path for the selected v8 components.**
- [ ] **Step 3: GREEN tests and organizer-image smoke.**
- [ ] **Step 4: Measure exact end-to-end runtime on the RTX 2060 runner and organizer image where available.** Keep serialization/tokenization/model/graph timings separately.
- [ ] **Step 5: If full scorer threatens the time budget, introduce measured cascade gating: cheap structured model on every row, expensive reranker only on the uncertainty/hardness subset. Tune coverage on OOF under a hard runtime constraint.**
- [ ] **Step 6: Produce final v8 ZIP, SHA256, artifact path and runtime report.**

### Task 8: Memory and evidence closure

**Files:**
- Modify: `ecup_matching/experiments/CURRENT.json`
- Modify/Create: `ecup_matching/experiments/v8/RESULTS.md`, `SAFE_METRICS.json`
- Modify: `docs/agent-memory/PROJECT_STATE.md`, `EXPERIMENT_INDEX.md`, `DECISIONS.md`

- [ ] **Step 1: Record every rejected as well as retained hypothesis, including prevalence diagnosis status and graph ablation.**
- [ ] **Step 2: Keep strict human OOF, test-like diagnostic, leaderboard score, and runtime in separate named fields.**
- [ ] **Step 3: Run `scripts/memory_policy.py` and repository test suite.**
- [ ] **Step 4: Final verification before merging/publishing: exact commit, artifact SHA, archive size, package smoke, output contract and runtime gate.**
