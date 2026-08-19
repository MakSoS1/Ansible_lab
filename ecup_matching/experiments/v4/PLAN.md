# E-CUP Matching — Iteration v4 Plan

Date: 2026-08-11
Status: **completed — retained via cross-fitted category routing**

## Original hypothesis

The retained v3 neural branch leaves quality on the table because it uses `rubert-tiny2`, only 180,000 human neural-training rows and no neural weak-label curriculum. The primary v4 engineering path therefore implemented a stronger Russian BERT cross-encoder trained on the complete leakage-safe human train, continued with confidence-filtered LLM soft labels, and optionally refined with hard-negative replay.

## Retained course correction

During v4 execution, the first home-RTX production attempt failed during host-memory-heavy preprocessing before any strong-encoder metric existed. That infrastructure failure was fixed in code and was never counted as a model score.

While the stronger-encoder branch was unavailable, a lower-risk candidate from the already-planned **blend selection** space was evaluated: preserve the immutable v3 structured/neural predictions and regularize category-specific neural alphas toward the global alpha.

To prevent category-alpha overfit, v4 added a stricter retention protocol than the original in-sample blend grid:

- 5-fold `GroupKFold`;
- groups are connected components of **all validation candidate edges**;
- a held-out item-component never tunes the alpha used to score itself;
- shrinkage prior is selected only from OOF Macro AP;
- after prior selection, final deployable alphas are fit on all labelled rows for the hidden-test submission.

This candidate produced an honest cross-fitted Macro AP `0.5276431099433088`, strictly above v3 `0.5254642645846543`, then passed the exact organizer-image offline package gate. It therefore became the retained v4 artifact.

The strong `ai-forever/ruBert-base` ladder remains implemented but is **not** part of the retained v4 evidence. Its CUDA/training requirements below apply to that future v4.1/v5 ablation only.

## Fixed baseline

Retained v3:

- model: v2b structured anchor + `cointegrated/rubert-tiny2` stage-1 global blend;
- structured/neural weights: `0.55 / 0.45`;
- Macro AP: `0.5254642645846543`;
- validation rows: `73,131`;
- train/validation item overlap: `0`;
- immutable submission SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`.

## Fixed data and split

- human outer train: 292,523 pairs before curriculum transformation;
- validation: 73,131 pairs;
- item overlap: exactly 0;
- split implementation remains the connected-component item-disjoint protocol.

For retained v4 routing selection, the frozen validation candidate graph contains 53,131 connected item-components and is cross-fitted by component.

## Retained v4 blend selection protocol

Inputs are immutable v3 validation predictions:

- structured score: retained `v2b-weak-curriculum`;
- neural score: retained v3 stage-1 `rubert-tiny2`;
- source v3 ZIP SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`;
- frozen prediction parquet SHA-256: `4112aa2556cb683ffca27cd9bd16c00a7149bb7e3279d1f2a6abb2b20438d643`.

Protocol:

1. inside each training fold, choose global neural alpha from the fixed alpha grid;
2. choose raw per-category alpha from the same fit fold;
3. shrink each category alpha toward the fold-global alpha;
4. evaluate each shrinkage prior only on held-out item-components;
5. choose the prior with highest OOF Macro AP;
6. after model-family/prior selection, fit final deployable category alphas on all labelled validation rows;
7. rebuild the immutable v3 ZIP by replacing routing coefficients only;
8. require exact organizer-image offline execution before private canonical freeze.

Selected prior: `4000`.

Headline retained score: `0.5276431099433088` cross-fitted Macro AP.

Final deployable coefficient fit: `0.5284493942551521` Macro AP; this larger full-fit value is not used as the unbiased headline.

## Original stronger-encoder implementation

Primary neural model: exact pinned `ai-forever/ruBert-base` revision `43be4261797042e172adf7476c558734f3cbb2a0`.

Implemented stages:

### v4a — full human stronger model

Train the stronger cross-encoder on the complete leakage-safe human curriculum; do not reproduce the v3 180k compaction.

### v4b — high-confidence weak curriculum

Warm-start from v4a with confidence-filtered LLM rows while authoritative human supervision remains dominant.

Weak confidence policy:

- `target <= 0.03` or `>= 0.97`: weight 1.0;
- `0.03 < target <= 0.15` or `0.85 <= target < 0.97`: weight 0.6;
- `0.15 < target <= 0.30` or `0.70 <= target < 0.85`: weight 0.3;
- `(0.30, 0.70)`: excluded.

### v4c — hard negatives with replay

Continue from the best parent using model-mined negatives with deterministic replay:

- 25% mined hard negatives;
- 25% positives/hard positives;
- 50% ordinary examples.

The code for these stages is retained for a future ablation, but no quality result from them is claimed in v4.

## Runtime/artifact rules

- Raw data, model weights and ZIPs remain private.
- Every binary winner is frozen under `submissions/vN/canonical/<sha256>/`.
- A package is not retained until exact organizer-image offline output succeeds.
- Existing canonical v3/v4 artifacts are immutable.

Retained v4 canonical artifact:

- SHA-256: `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`;
- bytes: `109,185,879`;
- private path: `submissions/v4/canonical/b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849/ecup-v4-submission.zip`.

## Retention criteria — final interpretation

Required for the retained blend-only v4 candidate:

- validation rows exactly 73,131;
- validation train overlap 0;
- routing hyperparameters selected out of fold by item-component;
- honest cross-fitted Macro AP strictly above v3;
- exact source hashes recorded;
- output continuous scores only;
- exact organizer-image offline execution succeeds with network disabled;
- output rows/order/schema/range/finite checks pass;
- ZIP <5 GB;
- canonical private artifact checksum/presence verified;
- public tests and memory policy pass.

The original CUDA-training requirement applies only if a newly trained strong encoder is proposed as the retained model. It is not applicable to v4's coefficient-only routing update because v4 reuses the already retained v3 learned weights unchanged.

## Decision

The cross-fitted category-routing candidate satisfied the final retention criteria and is the retained v4. See `RESULTS.md` for exact metrics, category alphas, package SHA and runtime evidence.