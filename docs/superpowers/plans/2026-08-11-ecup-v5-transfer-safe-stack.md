# E-CUP v5 Transfer-Safe Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct v5 evaluation semantics, evaluate a label-free fusion of the strongest genuinely held-out signals, and retain only improvements that remain honest on the immutable development split.

**Architecture:** Keep the frozen v5 split and direct specialist OOF files unchanged. Harden the official metric contract, then add a pure fixed-fusion module plus an aggregate runner that aligns private OOF artifacts by `row_index`/fold and evaluates predeclared non-learned fusion rules. If this does not reach 0.60, the next implementation cycle is a true outer-isolated nested stack rather than reuse of same-fold supervised OOF features.

**Tech Stack:** Python 3.11, NumPy, pandas, scikit-learn, PyArrow/Parquet, GitHub Actions, private Hugging Face artifacts.

## Global Constraints

- Branch: `ecup-matching-2026`; never modify/merge `main`.
- Split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Development rows: 285,210; sealed gold rows: 80,444; five component-disjoint folds; item overlap 0.
- Sealed-gold metrics/scores remain uninspected during development.
- Current clean direct anchor: explicit specialists `0.5683065131240066`.
- No target-fitted blender in the fixed-fusion experiment.
- All private data/OOF/model artifacts stay outside public Git.

---

### Task 1: Harden official Macro AP contract

**Files:**
- Modify: `ecup_matching/ml/metrics.py`
- Modify: `ecup_matching/ml/v5_evaluation.py`
- Modify: `ecup_matching/tests/test_metrics.py`
- Modify: `ecup_matching/tests/test_v5_evaluation.py`

**Interfaces:**
- Produces: `OFFICIAL_CATEGORIES: tuple[str, ...]`.
- Extends: `macro_average_precision(..., expected_categories=None, require_both_classes=False)`.
- Extends: `macro_ap_report(..., strict_official=False)`.

- [ ] **Step 1: Write failing tests**

Add tests proving strict mode rejects a missing official category and a category containing only one target class, while the existing generic toy-category API remains valid.

```python
with pytest.raises(ValueError, match="category set"):
    macro_average_precision(y, s, cats, expected_categories=OFFICIAL_CATEGORIES)

with pytest.raises(ValueError, match="both target classes"):
    macro_average_precision(y, s, cats, require_both_classes=True)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run in Actions: `python -m pytest ecup_matching/tests/test_metrics.py ecup_matching/tests/test_v5_evaluation.py -q`.
Expected: failure because the new strict arguments/constants do not exist.

- [ ] **Step 3: Implement minimal strict contract**

Validate exact category set only when `expected_categories` is provided; validate `{0,1}` per category only when `require_both_classes=True`. `macro_ap_report(strict_official=True)` passes both constraints using `OFFICIAL_CATEGORIES`.

- [ ] **Step 4: Verify focused + full suite GREEN**

Run: `python -m pytest ecup_matching/tests -q` and `python scripts/memory_policy.py`.
Expected: all tests pass and memory policy reports OK.

---

### Task 2: Add pure label-free fixed fusion

**Files:**
- Create: `ecup_matching/ml/v5_fixed_blend.py`
- Create: `ecup_matching/tests/test_v5_fixed_blend.py`

**Interfaces:**
- Produces: `percentile_rank(values: array-like) -> np.ndarray`.
- Produces: `fixed_blend_candidates(scores: Mapping[str, array-like], contrastive_cosine=None) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write failing tests**

Tests must prove rank transforms are finite, monotonic and bounded; candidate blends are symmetric with respect to source order; no function accepts/uses targets; and identical branch rankings remain identical after fusion.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest ecup_matching/tests/test_v5_fixed_blend.py -q`.
Expected: import/module failure because `v5_fixed_blend.py` does not exist.

- [ ] **Step 3: Implement minimal fusion rules**

Predeclared rules:

```text
prob_mean_4 = mean(category, weak, sparse, explicit)
rank_mean_4 = mean(rank(category), rank(weak), rank(sparse), rank(explicit))
rank_mean_3 = mean(rank(weak), rank(sparse), rank(explicit))
rank_mean_5 = rank_mean_4 plus rank(raw contrastive cosine) as fifth equal vote
```

Use average percentile ranks; clip probability scores to `[1e-6, 1-1e-6]`. No learned coefficients.

- [ ] **Step 4: Verify GREEN**

Run focused and full tests.

---

### Task 3: Run fixed-fusion OOF experiment

**Files:**
- Create: `ecup_matching/ml/run_v5_fixed_blend.py`
- Create: `.github/workflows/ecup-v5-fixed-blend.yml`
- Create/modify tests for runner alignment/coverage as needed.

**Interfaces:**
- Consumes aligned held-out OOF sources for category, weak, sparse, explicit and raw contrastive semantic features.
- Produces: `v5-fixed-blend-metrics.json` and `v5-fixed-blend-oof.parquet` in private HF storage.

- [ ] **Step 1: Write failing alignment/runner test**

Use small temporary parquet fixtures. Verify duplicate/missing `row_index` or fold mismatch raises before scoring.

- [ ] **Step 2: Verify RED**

Expected: runner does not yet exist.

- [ ] **Step 3: Implement runner**

Load frozen manifest; check split SHA; recover development rows/folds; align every source by exact `row_index`; ensure explicit/sparse/weak/category scores are finite; use raw `embedding_cosine` from held-out contrastive output; compute strict-official aggregate and held-fold reports. Explicit score is the comparison anchor.

The workflow downloads only public dataset inputs plus private OOF artifacts. It never opens/scans sealed-gold scores.

- [ ] **Step 4: Run workflow**

Report all four predeclared fusion scores, per-fold deltas vs explicit, and per-category AP. Keep a fusion only if aggregate AP exceeds `0.5683065131240066` and no fold regresses by more than `0.001`.

- [ ] **Step 5: Persist result and update memory**

Update `RESULTS.md`, `SAFE_METRICS.json`, `CURRENT.json` if current-best changes, `EXPERIMENT_INDEX.md`, `PROJECT_STATE.md`, and `DECISIONS.md`. Correct wording to: gold target labels were used once to stratify/freeze the split; gold metrics/scores have never been inspected during v5 development.

- [ ] **Step 6: Full verification/checkpoint**

Run full project tests + memory policy. Then verify hardened Memora ingest/checkpoint on GREEN state.

---

### Task 4: If fixed fusion remains below 0.60, start true outer-isolated stack

**Files:** separate follow-up implementation cycle after Task 3 evidence.

For each outer held fold `j`, every supervised feature used by the meta-model must be produced without using labels from `j`, including feature values for the meta-training rows. Reusing current same-fold OOF supervised score matrices is prohibited for the headline nested metric.

Order by cost: explicit+sparse -> +weak -> +contrastive -> pairwise Transformer teacher. Continue until honest development AP reaches 0.60 or every planned orthogonal branch has been exhausted, without opening sealed gold.
