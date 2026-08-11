# E-CUP v5 Target-Free Model/SKU Consistency Signal

Date: 2026-08-11

This experiment is declared while typed-consistency run `31497557239` is still in progress, before its metric is known.

## Anchor

Current verified clean target-free development OOF: `0.5952697490140912` from weak + sparse + old explicit + raw supervised contrastive + raw teacher2 equal global percentile ranks.

## Signal

Use the existing normalized `ItemNorm.model_codes` extracted from item name + attributes, with storage-capacity tokens excluded from model-code space because typed quantities handle them separately.

For an item pair:

- `+1` when both sides have model/SKU codes and their sets intersect;
- `-1` when both sides have at least one model/SKU code but the sets are disjoint;
- `0` when either side has no model/SKU code.

The score is deterministic, symmetric, label-free and bounded in `[-1, 1]`.

## Exactly one fusion candidate

`current5_plus_model_code_consistency` = equal global percentile ranks of the current five clean signals plus the deterministic model/SKU consistency score.

No threshold search, code weighting, category gating, learned coefficient or alternative variant is permitted after observing the metric.

## Gate

KEEP only if strict-official aggregate Macro AP improves `0.5952697490140912` and minimum held-fold delta vs anchor is at least `-0.001`. The engineering milestone is reached if aggregate dev OOF is `>= 0.60`. Sealed-gold metrics/scores remain uninspected.
