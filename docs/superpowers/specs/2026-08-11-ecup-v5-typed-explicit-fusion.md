# E-CUP v5 Typed-Explicit Fusion Gate

Date: 2026-08-11

This candidate set is frozen while typed-explicit outer-fold training run `31495661334` is still in progress and before its aggregate metric is known.

## Anchor

Current clean target-free five-signal OOF: `0.5952697490140912` from weak + sparse + old explicit + raw contrastive + raw teacher2 equal global percentile ranks.

## New source

`typed_explicit` is the direct held-out score from the same explicit-specialist HGB architecture retrained after typed quantity normalization and canonical attribute-value comparison. No model hyperparameters are changed.

## Exactly two predeclared fusion candidates

1. `current5_plus_typed_explicit`: equal global percentile ranks of weak + sparse + old explicit + contrastive + teacher2 + typed_explicit.
2. `current5_replace_explicit_with_typed`: equal global percentile ranks of weak + sparse + typed_explicit + contrastive + teacher2.

The second candidate tests whether typed representation is a strict replacement for old explicit instead of double-counting the explicit family. No other duplicated votes, source subsets, learned weights or category-specific coefficients are permitted in this gate.

## KEEP / milestone gate

A candidate is KEEP-eligible if strict-official aggregate Macro AP is above `0.5952697490140912` and minimum held-fold delta vs anchor is at least `-0.001`.

The engineering milestone is reached only when the aggregate development OOF is >= `0.60`, with `target_fitted_blender=false` and sealed-gold metrics/scores uninspected.
