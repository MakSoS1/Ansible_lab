# v7 Plan — identity-first full-context teacher

## Objective

Create a genuinely stronger pairwise signal instead of further tuning correlated lexical/meta layers. Stretch target: strict OOF Macro AP `>= 0.70`; minimum KEEP criterion: strict OOF must beat retained v5 `0.6018115534135564` without opening sealed gold and without violating runtime constraints.

## Frozen validation

- development rows: `285210`
- sealed gold rows: `80444`
- folds: `5`
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`
- cross-split item overlap: `0`
- metric: unweighted mean of `sklearn.metrics.average_precision_score` over exactly 20 official categories
- sealed gold metric opened: `false`

## Candidate A

`ruBERT-base` cross-encoder with:
- identity-first item serialization;
- canonical typed attributes before residual numeric noise;
- `max_length=256`;
- leakage-safe human + confidence-weighted weak curriculum;
- materially more exposure than the retained 800-step teacher2;
- five outer-fold held predictions only;
- progress/timing telemetry.

## Candidate B

Only if Candidate A is insufficient: aligned shared-key pair serialization as a second neural view, evaluated under the same outer-fold contract.

## Runtime

Preserve v6 prediction-preserving structured optimizations. Benchmark neural inference separately and then end-to-end. A full reference-pair run is required; a tiny smoke cannot prove the time gate.

## Status

IN PROGRESS — implementation starts with failing serializer/leakage/training-contract tests.