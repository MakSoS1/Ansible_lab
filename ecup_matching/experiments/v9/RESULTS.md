# v9 Results — leaderboard-adapted runtime-safe submission

Status: **COMPLETED — exact keeper passes public/private runtime gates**

Updated: 2026-08-12

## Final keeper

`ecup-v9-gate40-fp16-graph-0.5970059311-submission.zip`

- bytes: `1,251,659,961`;
- SHA-256: `925456cde1e47c50dc0141ce64bed5ef00d9f574152f285869ebea2db6935782`;
- build run: `31640050373`;
- release tag: `ecup-v9-gate40-final-eb2bcf18d53e`;
- structured workers capped at `8`;
- CUDA FP16/autocast;
- RTX batches: contrastive `256`, teacher `96`;
- sealed gold opened: `false`; gold rows scored: `0`.

The owner-reported v7 leaderboard score `~0.36` was retained only as external evidence (`used_for_fitting=false`). The requested `~0.5` region is a goal, not a claimed v9 leaderboard result. The actual v9 leaderboard score remains unknown until the platform scores this archive.

## Validation v2

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

Graph delta is positive on all five folds for both candidates. Gate40 was selected before final runtime because it dominates gate25 on strict OOF, graph OOF and target-stress mean. Gate25 was predeclared as the runtime fallback but was not needed.

## Production refit

Run `31639692541`:

- `15` targeted tests passed;
- all `285,210` development rows used;
- teacher fraction `~0.400025`;
- elapsed `74.4 s`;
- peak RAM `0.736 GiB`;
- artifact `9158411928`;
- private HF prefix `experiments/v9/production/gate40/853a3925ac2b`;
- sealed gold untouched.

Production refit is not validation; the numbers above remain the selection evidence.

## Why v8 was closed

The old gate70 evidence showed inner `run.py` around `731.22 s` but true outside-container wall `820.784 s`. The old workflow incorrectly wrote a pass because it checked process exit/output rather than enforcing measured outer wall. The platform then returned `Container did not finish in time` again.

v9 makes the outside-container wall authoritative.

## Runtime proof on RTX 2060 SUPER

First corrected private-size evidence run `31640233511` already showed:

- 275,000 rows;
- return code `0`;
- inner `run.py` `567.23 s`;
- outside wall `637.82083456 s`;
- complete output.

That workflow was red only because its validator incorrectly required graph-rescored numeric scores to lie in `[0,1]`. Diagnostic run `31641425359` confirmed exact schema, row count, ID order, finite values and nonconstant scores; only the artificial probability-range predicate failed. No clipping was added because clipping would create ties and alter the validated ranking.

Final independent dual gate run `31641656589` used the **same exact ZIP SHA** and organizer image:

| Gate | Rows | Acceptance | Outer wall | Headroom | Output | Result |
|---|---:|---:|---:|---:|---|---|
| public-size | `115,000` | `330 s` | **`281.821475323 s`** | **`48.178524677 s`** | valid | **PASS** |
| private-size | `275,000` | `700 s` | **`634.766220868 s`** | **`65.233779132 s`** | valid | **PASS** |

Both returned code `0`; public output had `108,272` unique scores and private output `271,386`. Final runtime evidence artifact: `9159596648`.

## Additional v9 quality experiments

Two essentially runtime-free meta ideas were tested leakage-free and rejected rather than added to the keeper:

- prevalence-weighted HGB: strict `-0.0000658754`, graph `-0.0000604780`, target-stress `-0.0000779612`;
- cross-fitted category-specific category/HGB fusion: strict `-0.0005091711`, graph `-0.0002712630`, target-stress `-0.0002826191`.

Therefore the final archive remains the already runtime-verified gate40+FP16+cap8+graph package; no post-runtime quality tweak changed its bytes.

## Final repository verification

Run `31642803187`:

- **423 passed**;
- **5 skipped**;
- `scripts/memory_policy.py`: **OK**.

## Final rules carried forward

- outside-container wall, not inner process time or exit status, is authoritative for timeout safety;
- a valid continuous ranking score is not required to be a calibrated `[0,1]` probability unless the competition contract explicitly says so;
- leaderboard, strict OOF and target-stress are separate evidence axes;
- a single leaderboard score is not used as row-level training/calibration data;
- sealed gold stays unopened;
- production refit is never reported as validation;
- do not weaken gates after observing a failure.

## Next external measurement

Upload the exact keeper ZIP and record the platform's measured v9 score separately. Until that happens, `~0.5` is a target and `0.4515676` is a target-stress diagnostic — neither is a claimed leaderboard AP.
