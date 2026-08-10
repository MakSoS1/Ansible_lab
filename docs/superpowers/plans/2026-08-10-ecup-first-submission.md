# E-CUP 2026 First Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the first fully valid E-CUP matching submission archive from a leakage-resistant human-label training pipeline, with reproducible training, offline inference, validation metrics, packaging, and CI artifacts.

**Architecture:** The first runnable submission is deliberately compact and CPU-friendly: deterministic item normalization + structured pair-similarity features + CatBoost binary ranking model. Validation uses connected-component/item-disjoint splits and the official Macro Average Precision objective. The runtime entrypoint reuses exactly the same feature code, loads a bundled CatBoost model, writes `id1,id2,predict`, and is packaged according to the organizer baseline contract discovered from the official lightweight archive. This v1 is the foundation for the already-selected hybrid cascade: later iterations add multilingual embeddings, weak labels, hard negatives, distillation, and an uncertainty Cross-Encoder without changing the submission contract.

**Tech Stack:** Python 3.11, pandas, pyarrow, numpy, rapidfuzz, catboost, scikit-learn metrics, GitHub Actions, Hugging Face private dataset.

## Global Constraints

- Modify only `MakSoS1/Ansible_lab` branch `ecup-matching-2026`.
- Do not modify existing Ansible files.
- Keep competition code under `ecup_matching/`; raw competition data and trained model binaries must not be committed to the public Git repository.
- Training data comes from private HF repo `Maksim123321/e-cup-2026-matching-private` using GitHub Actions secret `HF_TOKEN`.
- Metric is Macro Average Precision over the 20 categories using continuous scores.
- Validation must prevent any item ID from occurring in both train and validation.
- Inference must work without internet and accept organizer `items.parquet` and `matches.parquet` paths.
- The generated archive must contain all code/model/dependencies needed by the organizer runtime, stay far below the 5 GB archive limit, and produce `id1,id2,predict`.
- First submission prioritizes correctness, reproducibility and runtime headroom; neural additions are subsequent iterations after a valid measured v1 exists.

---

### Task 1: Inspect and freeze the organizer submission contract

**Files:**
- Create: `ecup_matching/BASELINE_CONTRACT.md`
- Create: `.github/workflows/ecup-inspect-baseline.yml`

**Produces:** exact archive tree, entry command/CLI flags, base image/dependencies, expected output location and file schema.

- [ ] Download only `baselines/matching-baseline-lightweight.zip` from the private HF repo on a GitHub Actions runner.
- [ ] Run `unzip -l`, extract text/config/source files, and print their contents while excluding binaries/secrets.
- [ ] Record the exact runtime contract in `BASELINE_CONTRACT.md`.
- [ ] Verify the full baseline archive has the same outer structure.

### Task 2: Leakage-free split and official metric

**Files:**
- Create: `ecup_matching/ml/split.py`
- Create: `ecup_matching/ml/metrics.py`
- Create: `ecup_matching/tests/test_split.py`
- Create: `ecup_matching/tests/test_metrics.py`

**Interfaces:**
- `component_split(matches: DataFrame, valid_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]`
- `macro_average_precision(y_true, y_score, categories) -> tuple[float, dict]`

- [ ] Write tests proving no item occurs in both folds, all rows are assigned once, split is deterministic, and Macro AP is the arithmetic mean of per-category `average_precision_score`.
- [ ] Run tests and confirm RED.
- [ ] Implement union-find connected components over `id1/id2`, deterministic component assignment stratified approximately by category/label mass, then rerun GREEN tests.

### Task 3: Shared item normalization and pair features

**Files:**
- Create: `ecup_matching/ml/textnorm.py`
- Create: `ecup_matching/ml/features.py`
- Create: `ecup_matching/tests/test_features.py`

**Produces:** a fixed ordered list `FEATURE_NAMES` and `build_pair_features(items, pairs) -> DataFrame` used identically by train and inference.

Required v1 signals:
- normalized-name exact equality/containment;
- RapidFuzz ratio, partial ratio, token-sort and token-set similarities;
- token Jaccard and character 3-gram Jaccard;
- name length ratio/difference;
- extracted integer/decimal set overlap and contradiction flags;
- alphanumeric model-code overlap/conflict;
- normalized quantity/unit overlap/conflict for common mass/volume/length/count units;
- parsed attribute key overlap, exact shared key-value agreement and contradiction rates;
- attribute value token Jaccard;
- category encoded as a categorical feature;
- missingness indicators.

- [ ] Write synthetic tests for exact duplicates, model-number conflicts (`A15` vs `A16`), quantity conflicts (`500 ml` vs `1 l`), and reordered equivalent attribute JSON.
- [ ] Implement cached per-item normalization so each item is parsed once, not once per pair.
- [ ] Verify deterministic feature order and finite numeric output.

### Task 4: Train/evaluate v1 CatBoost ranker

**Files:**
- Create: `ecup_matching/ml/train_v1.py`
- Create: `ecup_matching/ml/model_io.py`
- Create: `ecup_matching/tests/test_train_smoke.py`
- Create: `ecup_matching/requirements-ml.txt`

**CLI:**
`python -m ecup_matching.ml.train_v1 --items ... --matches ... --model-out ... --metrics-out ... --valid-pred-out ...`

Training policy:
- human labels only for v1;
- component/item-disjoint validation, target valid fraction 0.20;
- CatBoost `Logloss`, CPU, fixed seed 2026, depth 8, learning rate 0.05, up to 1200 iterations, early stopping 100;
- pair sample weights inversely proportional to category pair count so each category contributes approximately equally;
- choose best iteration on validation logloss but report official Macro AP overall + per-category;
- save model plus JSON manifest containing `FEATURE_NAMES`, seed, params, split statistics and validation metric.

- [ ] Write tiny smoke test that trains and predicts finite probabilities.
- [ ] Implement trainer and run full human training on GitHub Actions.
- [ ] Persist model/manifest/metrics only as private workflow artifacts and/or private HF artifacts, never Git blobs.

### Task 5: Offline inference entrypoint

**Files:**
- Create: `ecup_matching/submission/predict.py`
- Create: `ecup_matching/submission/run.py`
- Create: `ecup_matching/tests/test_submission_smoke.py`

**CLI contract:** adapted exactly to Task 1 baseline flags; internally it must read item and pair parquet, construct features, load bundled model, predict continuous scores, and write exactly columns `id1,id2,predict` in input pair order.

- [ ] Synthetic end-to-end test checks row/order preservation, finite scores in `[0,1]`, and exact output columns.
- [ ] Implement chunked inference (default 50k pairs) to bound memory.
- [ ] Add timing logs for load/feature/predict/write phases.

### Task 6: Build the first full submission archive

**Files:**
- Create: `ecup_matching/build_submission.py`
- Create: `ecup_matching/submission/requirements.txt` or organizer-compatible dependency bundle based on Task 1.
- Create: `.github/workflows/ecup-train-submit-v1.yml`

Workflow stages:
1. checkout + tests;
2. authenticate to HF without printing token;
3. download `items_human.parquet` + `matches.parquet` from private HF;
4. train v1 and write metrics;
5. run offline smoke inference on the validation subset;
6. download/extract organizer lightweight baseline template;
7. replace only solution implementation/model areas required by the frozen contract;
8. validate ZIP tree and size;
9. upload `ecup-v1-submission.zip`, `v1-metrics.json`, `v1-metrics.md`, validation predictions and manifest as GitHub Actions artifacts;
10. optionally mirror the submission archive/model into a private `submissions/v1/` path in the private HF dataset for durability;
11. cleanup raw data from runner in `if: always()`.

### Task 7: Verification gate

A v1 is complete only if all of the following are evidenced in Actions logs/artifacts:
- all unit/smoke tests pass;
- training finished and validation Macro AP is reported overall + all 20 categories;
- train/valid item intersection is exactly zero;
- no NaN/Inf scores;
- output schema/order test passes;
- offline inference works with network disabled after dependencies/model are present;
- submission ZIP structure matches the official baseline template;
- ZIP is well below 5 GB;
- model archive does not contain HF token, raw parquet, validation labels, or absolute runner paths.

## Iteration path after v1

1. Add 11M LLM soft labels with confidence/sample-weight curriculum.
2. Mine hard negatives inside category from high lexical/structured similarity conflicts.
3. Add cached multilingual bi-encoder embeddings per item and pair cosine/absolute-difference features.
4. Fine-tune a contrastive bi-encoder on human + filtered weak labels.
5. Distill an open-licensed teacher into the compact student.
6. Add tiny Cross-Encoder only for uncertainty band pairs.
7. Tune category-aware residual/meta-ranker and runtime Pareto frontier on H100 constraints.

## Self-review

- Spec coverage: first valid submit, first real training, leakage-free validation, official metric, offline packaging, reproducibility, artifacts and future hybrid path are covered.
- Placeholder scan: no TBD/TODO placeholders.
- Interface consistency: one shared feature builder is used by training and inference; one frozen CLI contract comes from the organizer baseline.
- Risk control: v1 does not depend on a GPU or internet at inference and keeps model size/runtime far below the organizer baseline ceiling.
