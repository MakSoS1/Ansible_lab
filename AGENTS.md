# AGENTS.md — READ THIS FIRST

Mandatory entry point for E-CUP matching work. Do not train, tune, inspect sealed-gold labels, change validation semantics, or publish another submission before reading the canonical state.

## Current iteration: v9 — completed

- Working branch: `ecup-v9-leaderboard-adapt`.
- Private artifact repo: `Maksim123321/e-cup-2026-matching-private`.
- Home GPU: private `MakSoS1/gpu-dispatch`, runner `ecup-rtx2060`, RTX 2060 SUPER 8 GiB.
- Immutable split: `285,210` development + `80,444` sealed gold, five component-disjoint folds, cross-split overlap `0`.
- Split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed gold remains unopened: `gold_metric_opened=false`, `gold_rows_scored=0`.
- Owner-reported v7 leaderboard anchor: `~0.36`, external only, `used_for_fitting=false`.
- Desired v9 leaderboard region: `~0.5`, target only; measured v9 leaderboard score is still unknown.

## Final v9 keeper

`ecup-v9-gate40-fp16-graph-0.5970059311-submission.zip`

- bytes `1,251,659,961`;
- SHA-256 `925456cde1e47c50dc0141ce64bed5ef00d9f574152f285869ebea2db6935782`;
- build run `31640050373`;
- release tag `ecup-v9-gate40-final-eb2bcf18d53e`;
- gate40 selective teacher;
- complete import-derived runtime closure;
- structured worker cap `8`;
- CUDA FP16/autocast;
- RTX batches: contrastive `256`, teacher `96`;
- frozen target-free graph: reciprocal bonuses `0`, endpoint rank `0.02`, ambiguity penalty `0.01`.

Do not alter these bytes and still call the result the verified v9 keeper.

## Validation v2

Source run `31639183423`.

| Candidate | Teacher fraction | Strict OOF | Fold-local graph OOF | Target-stress mean |
|---|---:|---:|---:|---:|
| gate25 | `0.2500227902` | `0.5947115591` | `0.5961903713` | `0.4507779206` |
| **gate40** | **`0.4000245433`** | **`0.5955054274`** | **`0.5970059311`** | **`0.4515676235`** |

Gate40 was selected before runtime; graph delta is positive on all five folds. The target-stress prevalence ratio is `0.566880890615799`. Target-stress is diagnostic and is not a leaderboard score.

Two near-zero-runtime meta alternatives were rejected on held-out evidence and are not in the keeper:

- fixed prevalence-weighted HGB: strict `-0.0000658754`, graph `-0.0000604780`, stress `-0.0000779612`;
- cross-fitted category fusion: strict `-0.0005091711`, graph `-0.0002712630`, stress `-0.0002826191`.

## Production refit

Run `31639692541`, artifact `9158411928`:

- all `285,210` development rows;
- teacher fraction `~0.400025`;
- `74.4 s`;
- peak RAM `0.736 GiB`;
- private HF `experiments/v9/production/gate40/853a3925ac2b`;
- sealed gold untouched.

Production refit is not validation.

## Final exact runtime proof

v8 is rejected: its inner `run.py` was around `731.22 s` but true outer wall was `820.784 s`; the old workflow incorrectly called it a pass. v9 therefore uses outside-container wall as the authoritative timeout criterion.

Final independent dual run `MakSoS1/gpu-dispatch#31641656589`, organizer image, exact keeper SHA:

| Gate | Rows | Acceptance | Outer wall | Headroom | Result |
|---|---:|---:|---:|---:|---|
| public-size | `115,000` | `330 s` | **`281.821475323 s`** | **`48.178524677 s`** | **PASS** |
| private-size | `275,000` | `700 s` | **`634.766220868 s`** | **`65.233779132 s`** | **PASS** |

Both returned `0`, preserved pair order, and produced valid finite nonconstant numeric scores. Evidence artifact `9159596648`. An independent earlier private run measured `637.82083456 s`, so the private runtime is reproducible.

Do not require ranking scores to be clipped to `[0,1]` unless the competition contract explicitly requires probabilities. The graph-rescored v9 output is a continuous ranking score; clipping would introduce ties and change validated ordering.

## Repository verification

Run `31642803187`:

- **423 passed**;
- **5 skipped**;
- `scripts/memory_policy.py`: **OK**.

## Mandatory reading order

1. `ecup_matching/experiments/CURRENT.json`
2. `docs/agent-memory/PROJECT_STATE.md`
3. `docs/agent-memory/EXPERIMENT_INDEX.md`
4. `ecup_matching/experiments/v9/PLAN.md`
5. `ecup_matching/experiments/v9/RESULTS.md`
6. `ecup_matching/experiments/v9/SAFE_METRICS.json`
7. `ecup_matching/experiments/v9/VALIDATION_V2.json`
8. `docs/agent-memory/DECISIONS.md`
9. `docs/agent-memory/SECURITY.md`
10. `docs/agent-memory/ITERATION_PROTOCOL.md`

## Binding invariants

- Never change the split to improve a metric.
- Never inspect/use sealed-gold labels/items for tuning, mining, calibration or runtime decisions.
- Every target-fitted development layer requires outer cross-fitting.
- Production refit is never validation.
- Leaderboard, strict OOF and target-stress remain separate evidence axes.
- A single leaderboard score is not converted into row-level labels or fitted calibration.
- Outside-container wall is authoritative for timeout safety.
- A smoke test is compatibility evidence, not runtime evidence.
- Runtime closure comes from `ecup_matching/ci/runtime_closure.py`, never a hand-maintained list.
- Never weaken quality/runtime gates after observing failure.

## Persistent memory

Hardened Memora pin: `bc64ff745a9b2c0e6245e0137654f041fba0c155`. Graph, LLM, external embeddings and auto-capture remain disabled. Raw competition data, model weights, submission ZIPs and the memory DB remain private.

Current checkpoint target: `--iteration v9`.

## Next external action

Submit the **exact** keeper archive above. When the platform finishes, record the measured v9 leaderboard score as a new external observation without rewriting local validation history.
