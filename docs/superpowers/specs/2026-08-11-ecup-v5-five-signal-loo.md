# E-CUP v5 Five-Signal Leave-One-Out Fusion

Date: 2026-08-11

## Entry evidence

The predeclared orthogonal rank-fusion experiment produced a new target-free development best:

- current clean anchor before teacher: `0.5870570848443828`;
- `current4_plus_teacher`: **`0.5952697490140912`**;
- delta: `+0.00821266416970845`;
- held-fold deltas: `+0.0051301`, `+0.0121087`, `+0.0101127`, `+0.0067881`, `+0.0133512`;
- target-fitted blender: false;
- sealed-gold metrics/scores: untouched.

The current five equal-rank signals are exactly:

1. weak specialist;
2. sparse TF-IDF specialist;
3. explicit attribute specialist;
4. supervised-contrastive raw cosine;
5. field-aware teacher2 raw score.

Gap to the engineering milestone `0.60` is `0.0047302509859088`.

## Rationale

A leave-one-signal-out analysis is a bounded robustness test of the already-selected five-signal ensemble. It answers whether any single vote dilutes the consensus after teacher2 is introduced. It does not fit coefficients or search continuous weights.

## Frozen candidate set

Evaluate exactly five equal-global-percentile-rank candidates, each removing one signal from the current five:

1. `loo_drop_weak`;
2. `loo_drop_sparse`;
3. `loo_drop_explicit`;
4. `loo_drop_contrastive`;
5. `loo_drop_teacher` (this must reproduce the prior `0.5870570848443828` anchor within numerical tolerance).

No pairwise subsets, duplicated votes, learned weights, category weights, metric-derived coefficients or post-result extra candidates are permitted in this ablation.

## Gate

Comparison anchor: `current4_plus_teacher = 0.5952697490140912`.

A candidate becomes the next development best only if:

- strict-official Macro AP > anchor;
- minimum held-fold delta vs anchor >= `-0.001`;
- target-fitted blender remains false;
- sealed-gold metrics/scores remain uninspected.

If all candidates stay below `0.60`, close subset pruning and move to genuinely new model evidence (typed numeric normalization / typed attributes). Do not begin arbitrary rank-weight sweeps on the same development rows.
