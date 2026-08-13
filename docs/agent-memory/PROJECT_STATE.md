# E-CUP Matching — Canonical Project State

Updated: 2026-08-13
Current iteration: **v9 compact — COMPLETED, published to private HF, awaiting platform score**

## Objective

Maximize E-CUP 2026 product-matching Macro AP with honest unseen-product validation and an offline organizer-compatible submission that finishes safely inside runtime limits.

The owner reports v7 as the best successfully scored submission so far at approximately `0.36`. That leaderboard value is external evidence only and was not used as a row-level label or fitted calibration target. The requested region near `0.5` remains a goal; compact v9 has no measured leaderboard score yet.

## Current keeper

Architecture: **gate40 + structured/meta stack + contrastive + selective teacher + frozen target-free graph**, with teacher/contrastive floating safetensors stored as FP16 to reduce package size.

Exact archive:

- `ecup-v9-compact-fp16storage-0.5970059311-submission.zip`;
- `596,925,132` bytes;
- SHA-256 `aabe663502b9dafe5b925347c3908d6bfe731045467aa85029da6255fbc78345`;
- build run `31675196422`;
- release tag `ecup-v9-compact-6ba133ce25f7`;
- private HF: `submissions/v9/compact/`;
- HF publication run `31677161875` verified both ZIP and `V9_COMPACT_KEEPER.json` remote paths.

The superseded 1.25 GB v9 (`925456c...35782`) received platform `Container did not finish in time`; retain it only as historical validation/runtime evidence and do not resubmit it.

## Immutable validation protocol

- human labels `365,654`;
- development rows `285,210`;
- sealed gold rows `80,444`;
- five component-disjoint folds;
- cross-split item/component overlap `0`;
- split SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- official metric: unweighted mean of `average_precision_score` over exactly 20 categories;
- sealed gold opened `false`, rows scored `0`.

## Validation v2

Run `31639183423`, target-stress prevalence ratio `0.566880890615799`.

| Candidate | Teacher fraction | Strict OOF | Fold-local graph OOF | Target-stress mean |
|---|---:|---:|---:|---:|
| gate25 | `0.2500227902` | `0.5947115591` | `0.5961903713` | `0.4507779206` |
| **gate40** | **`0.4000245433`** | **`0.5955054274`** | **`0.5970059311`** | **`0.4515676235`** |

These are the algorithm-selection metrics measured before compact storage. They were not re-labelled as compact OOF: production refit saw all development rows and sealed gold remains unopened. `0.4515676` is diagnostic, not a leaderboard prediction.

## Compact equivalence

RTX equivalence run `31675338174`, 20,000 identical pairs under organizer image:

- Spearman `0.999993145182`;
- Pearson `0.999993172228`;
- mean absolute delta `0.000163786536`;
- top-1% overlap `1.0`;
- top-5% / top-10% overlap `0.999`;
- exact schema/pair order and finite nonconstant outputs valid.

This is near-identical ranking evidence, not exact numeric equality and not a new OOF score.

## End-to-end runtime evidence

Run `MakSoS1/gpu-dispatch#31675903851` on NVIDIA GeForce RTX 2060 SUPER with `odsai/ecup26-matching-baseline:1.0`. Timer scope: safe ZIP extraction + docker inference + output validation.

| Gate | Rows | Total wall | Acceptance | Headroom | Result |
|---|---:|---:|---:|---:|---|
| public-size | `115,000` | **`293.569452608 s`** | `330 s` | `36.430547392 s` | **PASS** |
| private-size | `275,000` | **`646.947129008 s`** | `700 s` | `53.052870992 s` | **PASS** |

Extraction alone was `5.234229078 s` public and `5.714303003 s` private. Both returned code `0` and valid ordered output. Evidence artifact `9171929877`.

## Why package size changed

Immutable audit showed the old 1.25 GB archive was dominated by teacher (~680 MiB compressed) and contrastive (~449 MiB compressed) safetensors, not dead runtime files. FP16 storage reduced the exact keeper to `596,925,132` bytes, saving `654,734,829` bytes, while structured/meta models and inference logic stayed unchanged.

## Repository verification

Run `31676442849`:

- **425 passed**;
- **5 skipped**;
- `scripts/memory_policy.py`: **OK**.

## Binding lessons

- Infrastructure/runtime failures are not model scores.
- Production refit is not validation.
- Platform leaderboard, strict OOF and target-stress are separate evidence axes.
- Sealed gold is never opened to recover a leaderboard/runtime gap.
- Package extraction/setup belongs in organizer-like runtime measurement where feasible.
- Outside-container wall is authoritative for timeout safety.
- Mixed precision/storage compaction must have direct prediction/ranking evidence.
- Continuous ranking scores do not need clipping to `[0,1]` unless the contract requires probabilities.
- Never weaken a gate after observing failure.

## Current files to read

1. `ecup_matching/experiments/CURRENT.json`
2. `ecup_matching/experiments/v9/PLAN.md`
3. `ecup_matching/experiments/v9/RESULTS.md`
4. `ecup_matching/experiments/v9/SAFE_METRICS.json`
5. `docs/agent-memory/EXPERIMENT_INDEX.md`
6. `docs/agent-memory/DECISIONS.md`
7. `docs/agent-memory/SECURITY.md`
8. `docs/agent-memory/ITERATION_PROTOCOL.md`

## Next action

Submit the exact compact HF keeper to the competition platform. When it finishes scoring, record the measured leaderboard value separately without rewriting the frozen local validation evidence.
