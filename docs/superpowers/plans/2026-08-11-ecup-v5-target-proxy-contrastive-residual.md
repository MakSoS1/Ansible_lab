# E-CUP v5 Target-Proxy Contrastive Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and honestly validate a v5 product-matching system that treats hidden-best v2 as an immutable anchor, redesigns validation to match target difficulty better, adds an item-level contrastive representation and soft weak-label curriculum, and accepts `Macro AP >= 0.60` only on a frozen untouched gold holdout.

**Architecture:** v5 separates evaluation, representation learning and deployment. First create a balanced component-disjoint development-CV + frozen-gold manifest; then reproduce v2 on every fold; then add a regularized residual ranker, an item bi-encoder, larger soft weak-label curriculum and model-driven hard-example mining. Gold evaluation reads labels only after a candidate configuration hash is frozen.

**Tech Stack:** Python 3.12, pandas, NumPy, PyArrow, scikit-learn 1.9, PyTorch/Transformers available in the organizer image, pytest, GitHub Actions, private Hugging Face artifact storage, private RTX 2060 SUPER GPU dispatcher where available.

## Global Constraints

- Work only on `MakSoS1/Ansible_lab`, branch `ecup-matching-2026`, plus the already-approved private GPU dispatcher for trusted GPU execution.
- Raw competition products, pair rows, labels, weights and submission ZIPs stay private.
- `v2b-weak-curriculum` remains immutable production fallback and hidden-score anchor `0.2583231811423486`.
- Every validation split is item-component-disjoint; item overlap must equal zero.
- Frozen-gold labels may not influence model family, hyperparameters, encoder/checkpoint, weak weights, residual scale or category fitting.
- `honest_0p60` means Macro AP >= 0.600000 on a candidate frozen before the first gold-label scoring, with reproducible hashes.
- If honest 0.60 is not achieved, record the best honest result; never tune on gold and relabel it as honest.
- Organizer image/runtime contract remains `odsai/ecup26-matching-baseline:1.0` with `id1,id2,predict` output in input order.

---

### Task 1: Balanced grouped split and immutable gold manifest

**Files:**
- Create: `ecup_matching/ml/v5_validation.py`
- Create: `tests/test_v5_validation.py`
- Create: `ecup_matching/experiments/v5/PLAN.md`

**Interfaces:**
- Consumes: human pair frame containing `id1`, `id2`, `target`; optional precomputed split descriptors.
- Produces: `build_v5_split_manifest(matches, descriptors, *, gold_fraction=0.22, n_folds=5, seed=2026) -> dict`, `validate_manifest_no_overlap(matches, manifest) -> dict`, `manifest_sha256(manifest) -> str`.

- [ ] Write tests that construct multi-category connected components and assert zero item overlap among development folds and frozen gold.
- [ ] Write a RED test asserting the balanced assignment improves category/target/difficulty divergence versus the legacy size-only component split on a synthetic skewed graph.
- [ ] Implement deterministic component extraction and descriptor aggregation.
- [ ] Implement greedy multi-objective assignment to gold and five development folds using normalized absolute divergence over category, target and difficulty-bin counts.
- [ ] Serialize only component/row indices and aggregate statistics; never write product text or labels into public artifacts.
- [ ] Run `pytest tests/test_v5_validation.py -q` and commit.

### Task 2: Honest evaluation protocol and v2-relative reporting

**Files:**
- Create: `ecup_matching/ml/v5_evaluation.py`
- Create: `tests/test_v5_evaluation.py`

**Interfaces:**
- Produces: `macro_ap_report(frame, scores, *, category_col='category') -> dict`, `paired_component_bootstrap(...) -> dict`, `candidate_freeze_hash(config, prediction_sha, split_sha) -> str`, `assert_gold_evaluation_eligible(...)`.

- [ ] Write RED tests for macro AP, per-category AP, per-fold delta versus v2 and deterministic component bootstrap.
- [ ] Write RED tests proving gold evaluation refuses an unfrozen candidate or a split/config hash mismatch.
- [ ] Implement ordinary and target-weighted Macro AP without threshold calibration.
- [ ] Implement paired component bootstrap over v5-v2 AP differences.
- [ ] Implement candidate-freeze metadata and one-shot gold eligibility checks.
- [ ] Run focused tests and commit.

### Task 3: v5a validation audit on real data

**Files:**
- Create: `ecup_matching/ml/run_v5_validation_audit.py`
- Create: `.github/workflows/ecup-v5-validation-audit.yml`
- Modify after result: `ecup_matching/experiments/v5/RESULTS.md`

**Interfaces:**
- Reuses v2 training/runtime code and the v5 split manifest.
- Emits private `split-manifest.json`, fold v2 predictions, aggregate validation statistics and public-safe metrics JSON.

- [ ] Add tests for workflow/script argument validation and no-secret public outputs.
- [ ] On ephemeral CI/private data, construct the v5 split before any v5 training.
- [ ] Train/evaluate v2 independently on each development fold and once for the frozen-gold baseline path, but do not expose gold labels to later tuning jobs.
- [ ] Record fold variability, category AP and adversarial-slice baseline.
- [ ] Freeze the split manifest SHA in private storage and aggregate public docs.
- [ ] Do not advance if leakage checks fail.

### Task 4: Structured residual baseline v5b

**Files:**
- Create: `ecup_matching/ml/v5_residual.py`
- Create: `tests/test_v5_residual.py`
- Create: `ecup_matching/ml/run_v5_structured_residual.py`

**Interfaces:**
- `fit_residual_ranker(X, y, base_score, sample_weight=None, *, residual_strength, seed) -> model`
- `predict_residual(model, X, base_score) -> np.ndarray`

- [ ] RED test: zero residual model exactly preserves v2 ranking/scores within numerical tolerance.
- [ ] RED test: regularization limits correction magnitude on ambiguous synthetic data.
- [ ] Implement stable logit/sigmoid residual formulation with clipped base probabilities.
- [ ] Train only on development folds using v2 features plus category/conflict interactions; select residual strength by fold stability, not gold.
- [ ] Report mean/worst-fold delta and hard-slice deltas; freeze best v5b config.
- [ ] Commit only if tests pass.

### Task 5: Compact item serializer and contrastive pair representation

**Files:**
- Create: `ecup_matching/ml/v5_item_text.py`
- Create: `ecup_matching/ml/v5_embeddings.py`
- Create: `tests/test_v5_item_text.py`
- Create: `tests/test_v5_embeddings.py`

**Interfaces:**
- `serialize_item_v5(item_norm, *, max_chars=1200) -> str`
- `build_embedding_pair_features(emb_a, emb_b) -> np.ndarray`

- [ ] RED tests preserving model/SKU digits, units and category-relevant attributes while truncating deterministically.
- [ ] RED tests for symmetric cosine/L1/L2/product feature generation.
- [ ] Implement compact `[NAME]/[BRAND]/[MODEL]/[NUMERIC]/[ATTR]` serialization.
- [ ] Implement vectorized unique-item embedding cache and pair-feature lookup.
- [ ] Ensure pair representation is symmetric and deterministic.
- [ ] Run tests and commit.

### Task 6: Human-supervised contrastive encoder v5c

**Files:**
- Create: `ecup_matching/ml/train_v5_biencoder.py`
- Create: `tests/test_train_v5_biencoder.py`
- Create: private trusted GPU profile/workflow only if the local runner is online; otherwise use a verified alternative GPU path without changing data policy.

**Interfaces:**
- Inputs: development-only human pairs, split manifest, local pretrained encoder directory.
- Outputs: encoder checkpoint, OOF item embeddings/pair scores, metrics and hashes.

- [ ] RED unit tests for batch construction, no-gold-item filtering and contrastive/ranking loss behavior.
- [ ] Start with a small open Russian/multilingual encoder that fits 8GB VRAM; benchmark before considering a larger teacher.
- [ ] Train one fold at a time so every development prediction is OOF.
- [ ] Combine supervised contrastive objective with pair ranking/BCE supervision.
- [ ] Add compact embedding features to the residual ranker; choose candidate only from five-fold development results.
- [ ] Record whether every fold improves or where it regresses.

### Task 7: Target-proxy/domain-shift audit

**Files:**
- Create: `ecup_matching/ml/v5_target_proxy.py`
- Create: `tests/test_v5_target_proxy.py`

**Interfaces:**
- `fit_domain_proxy(human_desc, target_desc, groups, seed) -> oof_probabilities, model`
- `density_ratio_weights(prob, *, clip=(0.25, 4.0)) -> np.ndarray`

- [ ] RED tests for OOF-only domain probabilities and bounded density ratios.
- [ ] Compute label-free descriptors for real unlabeled organizer-like candidate pairs when a representative sample is available.
- [ ] Quantify AUC/domain separability; high AUC is explicit evidence the old validation distribution differs.
- [ ] Produce target-weighted development metrics as supplemental selection evidence.
- [ ] Never fabricate proxy data when real target-like candidates are unavailable.

### Task 8: Multi-million soft weak-label curriculum v5d

**Files:**
- Create: `ecup_matching/ml/v5_weak_stream.py`
- Create: `tests/test_v5_weak_stream.py`
- Modify: `ecup_matching/ml/train_v5_biencoder.py`

**Interfaces:**
- Streaming selector accepts weak parquet, forbidden item IDs, category quotas and optional target-proxy weights.
- Emits bounded-memory training shards with soft `target` and `source_weight`.

- [ ] RED tests that all frozen-gold/development-evaluation items are excluded from weak training for the relevant fold.
- [ ] RED tests that probabilities remain soft and deterministic sampling respects category/class quotas.
- [ ] Stream 2-4M weak rows if resource limits permit; no full 11M pandas materialization.
- [ ] Compare weak curriculum against v5c on exactly the same development folds.
- [ ] Keep only if gain is stable and target-weighted evidence does not contradict ordinary CV.

### Task 9: Model-driven hard-example continuation v5e

**Files:**
- Create: `ecup_matching/ml/v5_hard_mining.py`
- Create: `tests/test_v5_hard_mining.py`

**Interfaces:**
- `mine_hard_examples(weak_scores, teacher_targets, v2_scores, descriptors, *, limits, seed) -> DataFrame`

- [ ] RED tests for deterministic disagreement mining and mandatory easy-example replay.
- [ ] Stream-score weak pool with v2 and v5d; collect teacher/model and v2/v5 disagreements.
- [ ] Prioritize SKU/model/number/quantity/revision contradictions without making them the entire training set.
- [ ] Continue training with a bounded replay mix.
- [ ] Retain only if five-fold evidence improves.

### Task 10: Freeze best candidate and one-shot gold evaluation

**Files:**
- Create: `ecup_matching/ml/run_v5_gold_evaluation.py`
- Create: `.github/workflows/ecup-v5-gold-evaluate.yml`
- Modify: `ecup_matching/experiments/v5/RESULTS.md`

**Interfaces:**
- Inputs: immutable split SHA, candidate config SHA, checkpoint SHA, OOF/deployment artifacts.
- Outputs: frozen-gold v2/v5 Macro AP, per-category AP, paired component bootstrap CI, `honest_0p60` boolean.

- [ ] Freeze candidate config/checkpoint/preprocessing hashes before reading gold labels.
- [ ] Enforce one-shot eligibility checks in code.
- [ ] Score v2 and v5 on identical gold rows.
- [ ] Set `honest_0p60=true` only when Macro AP >= 0.600000 and every honesty assertion passes.
- [ ] If score is below 0.60, record it permanently; do not use gold to tune the same v5 generation.

### Task 11: If honest gold <0.60, continue with new nested development generations without reusing gold for tuning

**Files:**
- Modify/create only after diagnostic evidence identifies the limiting error family.

**Interfaces:**
- New candidate generations may use development CV and weak/domain evidence, but not gold labels.

- [ ] Diagnose fold-consistent errors by adversarial slice and category.
- [ ] Test one architectural hypothesis at a time: better item encoder, category-specific residual interactions, better weak selection, or offline teacher distillation.
- [ ] Require development-fold consistency before creating a new frozen candidate generation.
- [ ] A later candidate may be scored on a newly rotated untouched holdout only; do not repeatedly optimize against the original gold score.

### Task 12: Organizer package and current-state update

**Files:**
- Create: `ecup_matching/submission/predict_v5.py`
- Create: `ecup_matching/build_submission_v5.py`
- Create: `tests/test_v5_submission.py`
- Modify: `ecup_matching/experiments/CURRENT.json`
- Modify: `docs/agent-memory/PROJECT_STATE.md`
- Modify: `docs/agent-memory/EXPERIMENT_INDEX.md`
- Modify: `docs/agent-memory/DECISIONS.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Organizer CLI unchanged: `--output_path`, `--items_path`, `--matches_path`.

- [ ] RED tests for exact row order/schema/finite score/range and no network dependency.
- [ ] Package v5 with immutable artifact hashes.
- [ ] Run exact organizer image with `--network none`, read-only mounts/root, and representative neural routing.
- [ ] Benchmark against 780-second private limit and require >=25% safety margin where representative timing is available.
- [ ] Update CURRENT only if v5 passes the replacement gate; otherwise keep v2 as production-best and document v5 honestly.
- [ ] Run full pytest + memory/documentation policy + Memora checkpoint workflow before completion claim.
