# E-CUP v5 Target-Free Typed Consistency Signal

Date: 2026-08-11

## Anchor

Current clean target-free five-signal development OOF: `0.5952697490140912`.

## New signal

For each item pair, use canonical quantities extracted from name + attributes. Group values by typed dimension. For every dimension observed on both sides:

- +1 if the canonical value sets intersect;
- -1 if both sides contain the dimension but values conflict;
- missing-on-one-side contributes 0 and is not comparable.

The pair score is `(equal_dimensions - conflicting_dimensions) / comparable_dimensions`, or 0 when there are no comparable dimensions. It is deterministic, label-free, symmetric and bounded in [-1, 1].

This includes existing mass/volume/length/count plus the newly added storage, battery, power, voltage, frequency and display-diagonal dimensions.

## Exactly one fusion candidate

`current5_plus_typed_consistency` = equal global percentile ranks of the current five signals plus the deterministic typed-consistency score.

No learned coefficient, threshold, category weight, conflict penalty search or additional variant is allowed after observing the result.

## Gate

KEEP only if strict-official Macro AP improves `0.5952697490140912` and minimum held-fold delta is >= `-0.001`. Milestone is reached only if aggregate dev OOF >= `0.60`. Sealed-gold metrics/scores remain uninspected.
