# v9 Results — compact runtime-hardened submission

Status: **COMPLETED — compact keeper published to private HF and passes end-to-end public/private runtime gates**

Updated: 2026-08-13

## Current submission keeper

`ecup-v9-compact-fp16storage-0.5970059311-submission.zip`

- bytes: `596,925,132` (~569 MiB);
- SHA-256: `aabe663502b9dafe5b925347c3908d6bfe731045467aa85029da6255fbc78345`;
- build run: `31675196422`;
- release tag: `ecup-v9-compact-6ba133ce25f7`;
- private HF path: `submissions/v9/compact/ecup-v9-compact-fp16storage-0.5970059311-submission.zip`;
- HF publication run: `31677161875`;
- sealed gold opened: `false`; gold rows scored: `0`.

The previous 1.25 GB v9 archive (`925456c...35782`) returned `Container did not finish in time` on the platform and is now retained only as historical validation/runtime evidence. It is rejected for further submission because of packaging/runtime risk.

## Validation v2 — algorithm selection evidence

Frozen protocol:

- development rows `285,210`;
- sealed gold rows `80,444`;
- five component-disjoint folds;
- cross-split item overlap `0`;
- split SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- official metric: unweighted Macro AP over exactly 20 categories.

Validation run `31639183423`; frozen target-stress prevalence ratio `0.566880890615799`; graph config: reciprocal bonuses `0`, endpoint-rank `0.02`, ambiguity penalty `0.01`.

| Candidate | Teacher fraction | Strict OOF | Fold-local graph OOF | Target-stress mean |
|---|---:|---:|---:|---:|
| gate25 | `0.2500227902` | `0.5947115591` | `0.5961903713` | `0.4507779206` |
| **gate40** | **`0.4000245433`** | **`0.5955054274`** | **`0.5970059311`** | **`0.4515676235`** |

Graph delta is positive on all five folds for gate40. Gate40 was selected before final runtime and dominates gate25 on strict OOF, graph OOF and target-stress mean.

The compact archive changes checkpoint storage precision after the production refit, so the table above is **inherited algorithm-selection evidence**, not a newly measured compact OOF score. Re-running strict OOF with the production refit would be invalid because that refit already saw all development rows; sealed gold remains unopened.

`0.4515676` remains a target-stress diagnostic, not a claimed leaderboard AP. The owner-reported v7 leaderboard `~0.36` remains an external anchor only (`used_for_fitting=false`). Actual compact v9 leaderboard score is still unknown until the platform scores it.

## Production refit

Run `31639692541`:

- all `285,210` development rows used;
- teacher fraction `~0.400025`;
- elapsed `74.4 s`;
- peak RAM `0.736 GiB`;
- artifact `9158411928`;
- private HF prefix `experiments/v9/production/gate40/853a3925ac2b`;
- sealed gold untouched.

Production refit is not validation.

## Why the 1.25 GB v9 was superseded

The previous v9 archive was `1,251,659,961` bytes. Its old hard gate started the timer **after ZIP extraction**, so it did not test the full setup cost that may be counted by the organizer. After another platform `Container did not finish in time`, package size/setup became the strongest packaging hypothesis, though it is not claimed as proven platform root cause.

An immutable ZIP audit showed nearly all compressed bytes were model weights:

- teacher safetensors ~680 MiB;
- contrastive safetensors ~449 MiB;
- structured joblib ~43 MiB;
- runtime code/metadata negligible.

Dead-file removal could not materially reduce size. The compact builder therefore stores floating tensors in the teacher and contrastive safetensors as IEEE FP16 while preserving integer tensors and safetensors metadata. Structured/meta models and runtime logic remain unchanged; dead Python bytecode is removed.

Result: `596,925,132` bytes, saving `654,734,829` bytes (size ratio `0.4769067883`).

## Original-vs-compact equivalence on RTX 2060 SUPER

Run `31675338174`, same organizer image and fixed 20,000-pair fixture:

- original runtime: `91.352614853 s`;
- compact runtime: `81.015456088 s`;
- mean absolute prediction delta: `0.000163786536`;
- p99 absolute delta: `0.001901140684`;
- p999 absolute delta: `0.008163265306`;
- maximum absolute delta: `0.100742311771`;
- Pearson: `0.999993172228`;
- Spearman rank correlation: `0.999993145182`;
- top-1% overlap: `1.0000`;
- top-5% overlap: `0.9990`;
- top-10% overlap: `0.9990`;
- top-25% overlap: `0.9996`;
- exact row count/pair order and finite nonconstant outputs: valid.

This supports near-identical ranking transfer; it is deliberately **not** described as exact numeric equality or as a new OOF measurement.

## End-to-end runtime proof including ZIP extraction

Final run `MakSoS1/gpu-dispatch#31675903851`, organizer image `odsai/ecup26-matching-baseline:1.0`, NVIDIA GeForce RTX 2060 SUPER. The timer starts before safe ZIP extraction and ends after output validation. Network download and fixture preparation are excluded.

| Gate | Rows | Extraction | Inference | Validation | Total | Acceptance | Headroom | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| public-size | `115,000` | `5.234229078 s` | `287.592726358 s` | `0.738098444 s` | **`293.569452608 s`** | `330 s` | **`36.430547392 s`** | **PASS** |
| private-size | `275,000` | `5.714303003 s` | `640.236212537 s` | `0.992280967 s` | **`646.947129008 s`** | `700 s` | **`53.052870992 s`** | **PASS** |

Both returned code `0`, exact pair order, valid finite nonconstant outputs. Unique scores: public `108,264`, private `271,298`. Evidence artifact `9171929877`.

## HF publication proof

Publication run `31677161875` re-downloaded the release asset, verified exact size and SHA, then uploaded and re-listed both paths in private HF:

- `submissions/v9/compact/ecup-v9-compact-fp16storage-0.5970059311-submission.zip`;
- `submissions/v9/compact/V9_COMPACT_KEEPER.json`.

The manifest keeps `leaderboard_score=null` until the platform returns a measured result.

## Additional v9 quality experiments

Two essentially runtime-free meta ideas were tested leakage-free and rejected rather than added:

- prevalence-weighted HGB: strict `-0.0000658754`, graph `-0.0000604780`, target-stress `-0.0000779612`;
- cross-fitted category-specific category/HGB fusion: strict `-0.0005091711`, graph `-0.0002712630`, target-stress `-0.0002826191`.

## Repository verification

Run `31676442849`:

- **425 passed**;
- **5 skipped**;
- `scripts/memory_policy.py`: **OK**.

## Binding rules carried forward

- organizer-like wall must include ZIP extraction/setup where feasible;
- outside-container wall, not inner process time or exit status alone, is authoritative for timeout safety;
- packaging changes that affect numeric inference require direct prediction/ranking evidence;
- leaderboard, strict OOF and target-stress are separate evidence axes;
- a single leaderboard score is not used as row-level training/calibration data;
- sealed gold stays unopened;
- production refit is never reported as validation;
- continuous ranking scores need not be clipped to `[0,1]` unless the competition contract explicitly requires calibrated probabilities;
- do not weaken runtime or validation gates after observing failure.

## Next external measurement

Submit the exact compact keeper. After platform scoring, record the measured v9 leaderboard score separately without rewriting local validation history.
