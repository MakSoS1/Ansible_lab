# E-CUP Matching — Experiment Index

This file is the short registry. Detailed metrics live in each experiment's `RESULTS.md`.

| Version | Status | Model / idea | Validation | Macro AP | Runtime | Private artifact | Next decision |
|---|---|---|---|---:|---:|---|---|
| v1 | completed | structured/lexical HGB | item-disjoint connected-component split, 73,131 pairs | 0.4961654895 | 308.57 s train/eval; 1.78 s / 1k-pair organizer smoke | `submissions/v1/ecup-v1-submission.zip` | superseded by v2b |
| v2 | completed | 2024-inspired product features + confidence-filtered LLM weak labels (`v2b-weak-curriculum`) | exact same item-disjoint validation, 73,131 pairs, 0 shared items | **0.5010008995** | structured train/ablations ~953 s; **334 s / 275k-pair offline organizer benchmark**; 446 s (57.18%) headroom to 780 s private limit | `submissions/v2/ecup-v2-submission.zip` | v3: model-driven reranker/hard negatives once GPU Studio is accessible |

## v2 ablation headline

- v2a, human + 2024-inspired product-aware features: `0.5006971263` Macro AP — accepted.
- v2b, v2a + 300k confidence-filtered weak labels: `0.5010008995` — retained.
- v2c, naive static hard-negative reweighting: `0.4957263069` — rejected.
- Lightning neural reranker code is implemented, but the authenticated account exposed no reusable Studio and denied Studio creation with HTTP 403, so no neural metric was fabricated and no GPU credits were consumed.

## Required interpretation

- Headline scores are comparable because v1/v2 use the identical leakage-resistant human validation split.
- `REVIEW` or threshold accuracy is irrelevant here; the competition metric is ranking Average Precision by category.
- Private artifacts never belong in this public Git repository.
