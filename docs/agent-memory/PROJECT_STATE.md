# E-CUP Matching — Canonical Project State

Updated: 2026-08-12
Current iteration: **v9 — COMPLETED, awaiting platform score**

## Objective

Maximize E-CUP 2026 product-matching Macro AP with honest unseen-product validation and an offline organizer-compatible submission that finishes safely inside runtime limits.

The owner reports v7 as the best successfully scored submission so far at approximately `0.36`. That leaderboard value is external evidence only: it was not used as a row-level label or fitted calibration target. The requested v9 region near `0.5` remains a goal; the measured v9 leaderboard score is unknown until the platform scores the final archive.

## Final v9 keeper

Architecture: **gate40 + complete optimized runtime closure + structured cap8 + CUDA FP16 + frozen target-free graph postprocess**.

Exact archive:

- `ecup-v9-gate40-fp16-graph-0.5970059311-submission.zip`;
- `1,251,659,961` bytes;
- SHA-256 `925456cde1e47c50dc0141ce64bed5ef00d9f574152f285869ebea2db6935782`;
- build run `31640050373`;
- release tag `ecup-v9-gate40-final-eb2bcf18d53e`.

Runtime semantics:

- structured workers capped at `8`;
- CUDA FP16/autocast;
- contrastive batch `256`, teacher batch `96` on RTX 2060 SUPER;
- fold-validated graph config: reciprocal bonuses `0`, endpoint-rank weight `0.02`, ambiguity penalty `0.01`.

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

All five folds have positive graph delta for gate40. Gate40 was selected before final runtime and dominates gate25 on the three frozen quality/stress axes. Gate25 was predeclared and production-refit as a fallback but was not activated.

The `0.4515676` target-stress number is diagnostic, not a claimed leaderboard prediction.

## Production refit

Run `31639692541`:

- all `285,210` development rows;
- teacher fraction `~0.400025`;
- `74.4 s`;
- peak RAM `0.736 GiB`;
- artifact `9158411928`;
- private HF `experiments/v9/production/gate40/853a3925ac2b`;
- sealed gold untouched.

Production refit is never treated as validation.

## Why v8 was closed

v8 gate70 evidence showed inner `run.py` around `731.22 s` but true outside-container wall `820.784 s`. The old workflow nevertheless wrote a pass because it checked process exit/output instead of enforcing outer wall. The platform then returned `Container did not finish in time` again.

Binding correction: **outside-container wall is authoritative**.

## Exact runtime evidence

A first corrected 275k run `31640233511` completed the exact v9 bytes in `637.82083456 s` outside wall. Its red status was traced to an artificial validator requirement that graph-rescored scores be probabilities in `[0,1]`; diagnostic run `31641425359` proved exact schema, row count, ID order, finite and nonconstant numeric scores. No clipping was added because clipping would alter ranking through ties.

Final independent dual run `MakSoS1/gpu-dispatch#31641656589` on RTX 2060 SUPER and organizer image:

| Gate | Rows | Acceptance | Outside wall | Headroom | Result |
|---|---:|---:|---:|---:|---|
| public-size | `115,000` | `330 s` | **`281.821475323 s`** | **`48.178524677 s`** | **PASS** |
| private-size | `275,000` | `700 s` | **`634.766220868 s`** | **`65.233779132 s`** | **PASS** |

Both returned code `0`, preserved exact pair order, and produced valid finite nonconstant scores. Evidence artifact `9159596648`.

## Rejected v9 meta experiments

Two essentially runtime-free alternatives were tested leakage-free and rejected:

- fixed prevalence-weighted HGB: strict `-0.0000658754`, graph `-0.0000604780`, stress `-0.0000779612`;
- cross-fitted category-specific category/HGB fusion: strict `-0.0005091711`, graph `-0.0002712630`, stress `-0.0002826191`.

Neither changes the keeper bytes.

## Repository verification

Run `31642803187`:

- **423 passed**;
- **5 skipped**;
- `scripts/memory_policy.py`: **OK**.

## Binding lessons

- Infrastructure/OOM/API failures are not model scores.
- Production refit is not validation.
- Platform leaderboard, strict OOF and target-stress are separate evidence axes.
- A single leaderboard score is not converted into row-level labels or direct calibration.
- Sealed gold is never opened to recover a leaderboard/runtime gap.
- High single-fold research diagnostics do not substitute for complete strict OOF.
- Outside-container wall, not inner process time or exit code, is authoritative for timeout safety.
- A continuous ranking score need not lie in `[0,1]` unless the competition contract explicitly requires a probability.
- Final runtime closure is derived from imports, never hand-maintained.
- Mixed precision is retained only with quality/ranking evidence.
- Never weaken a gate after observing failure.

## Current files to read

1. `ecup_matching/experiments/CURRENT.json`
2. `ecup_matching/experiments/v9/PLAN.md`
3. `ecup_matching/experiments/v9/RESULTS.md`
4. `ecup_matching/experiments/v9/SAFE_METRICS.json`
5. `ecup_matching/experiments/v9/VALIDATION_V2.json`
6. `docs/agent-memory/EXPERIMENT_INDEX.md`
7. `docs/agent-memory/DECISIONS.md`
8. `docs/agent-memory/SECURITY.md`
9. `docs/agent-memory/ITERATION_PROTOCOL.md`

## Next action

Run hardened Memora ingest/checkpoint for the completed v9 state, then submit the exact keeper ZIP. After the platform scores it, record that leaderboard number separately without rewriting local validation history.
