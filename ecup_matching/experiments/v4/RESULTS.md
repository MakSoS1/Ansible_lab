# E-CUP Matching — Iteration v4 Results

Date: 2026-08-11
Status: **in progress**

## Baseline

Retained v3 Macro AP: `0.5254642645846543` on the fixed 73,131-row item-disjoint validation with zero train/validation item overlap.

Immutable v3 submission SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`.

## Planned measured candidates

- v4a: `ai-forever/ruBert-base` + complete human curriculum.
- v4b: v4a continuation + confidence-filtered LLM weak curriculum.
- v4c: best v4a/v4b continuation + model-mined hard negatives with 50% ordinary replay.

No v4 metric is recorded here until a real run has produced and passed validation checks.

## Execution evidence

No production v4 training run retained yet.

## Current decision

v3 remains current best until a v4 candidate strictly exceeds `0.5254642645846543` and passes the packaging/runtime gates.
