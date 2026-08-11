# E-CUP v5 Fixed Category-Shrunk Simplex Gate

Date: 2026-08-11

## Frozen entry state

- six-signal equal-rank OOF Macro AP: `0.5975445721449741`;
- fully outer-cross-fitted global simplex OOF Macro AP: `0.5992720660193247`;
- immutable v5 split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- sealed gold remains unopened and has `0` scored rows.

## Frozen candidate

Fit one category-shrunk simplex candidate with these fixed choices, declared before its real OOF result is inspected:

- signals: the same six frozen v5 signals;
- target transform: none; each signal is converted only to a global target-free percentile rank;
- outer folds: the immutable five v5 folds;
- global coefficients: nonnegative simplex weights fitted only on the four outer-train folds by direct Macro AP coordinate ascent;
- local coefficients: one nonnegative simplex per official category, fitted only on that category's rows inside the same four outer-train folds;
- local initialization: the outer-train global simplex weights;
- shrinkage: `(support * local + 8000 * global) / (support + 8000)`;
- prior strength: exactly `8000`, frozen from the previously used v4 conservative shrinkage range; it must not be changed after observing this candidate's v5 OOF result;
- prediction: each excluded outer-fold row is scored only with coefficients fitted without any target from that outer fold.

No prior grid, category-specific prior, negative weights, threshold tuning, post-result source subset, or post-result coefficient search is allowed in this gate.

## Gate

KEEP only if strict-official development Macro AP is greater than `0.5992720660193247` and the sealed gold remains unopened. The engineering milestone is reached at strict-official OOF Macro AP `>= 0.60`.
