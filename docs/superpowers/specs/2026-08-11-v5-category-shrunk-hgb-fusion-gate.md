# E-CUP v5 Category-Shrunk + HGB Equal-Rank Fusion Gate

Date: 2026-08-11

## Frozen entry state

- fixed category-shrunk simplex strict outer-OOF Macro AP: `0.60095424180184`;
- fixed nonlinear HGB meta-stack strict outer-OOF Macro AP: `0.6006290884983169`;
- immutable split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- sealed gold remains unopened and has `0` scored rows.

Both component vectors are fully outer-cross-fitted and use the same immutable development rows/folds. Their modeling biases differ: one optimizes direct Macro AP with fixed category shrinkage; the other uses a fixed shallow nonlinear HGB over the six target-free rank signals plus category.

## Frozen candidate

Before inspecting the fusion metric, evaluate exactly one label-free combination:

`equal_rank = 0.5 * percentile_rank(category_shrunk_oof) + 0.5 * percentile_rank(hgb_stack_oof)`

No alternative weights, source subsets, category weights, thresholds, calibration, or post-result blending are authorized in this gate.

## Gate

KEEP only if strict-official development Macro AP exceeds `0.60095424180184` while sealed gold remains unopened. Otherwise retain category-shrunk simplex as v5 production best and close this meta-fusion branch.
