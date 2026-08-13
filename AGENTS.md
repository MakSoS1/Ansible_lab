# AGENTS.md — READ THIS FIRST

Mandatory entry point for E-CUP matching work. Do not train, tune, inspect sealed-gold labels, change validation semantics, or publish another submission before reading the canonical state.

## Current iteration: v10 — completed keeper

- Working branch: `ecup-v10-distilled-fast`.
- Private artifact repo: `Maksim123321/e-cup-2026-matching-private`.
- Home GPU: private `MakSoS1/gpu-dispatch`, runner `ecup-rtx2060`, NVIDIA GeForce RTX 2060 SUPER.
- Immutable split: `285210` development + `80444` sealed gold, five component-disjoint folds, cross-split overlap `0`.
- Split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed gold remains unopened: `gold_metric_opened=false`, `gold_rows_scored=0`.
- Owner-reported v7 leaderboard anchor: `~0.36`, external only, `used_for_fitting=false`.
- Desired v10 leaderboard region: `~0.5`, target only; measured v10 leaderboard score is unknown until platform scoring.

## Final v10 keeper

`ecup-v10-no-teacher-graph-0.5950413763-submission.zip`

- bytes `480249520`;
- SHA-256 `6cebc276f45fc52247db054eb83d2a8110b25d4407cc34b0d5b148a4773c321d`;
- build run `31689478925`;
- release tag `ecup-v10-faststack-9de2bc83f878`;
- source SHA `9de2bc83f878c87703c3290670f042bfdbb70dfc`;
- private HF `submissions/v10/final/`;
- teacher checkpoint packaged `false`;
- CPU structured and GPU contrastive branches overlap;
- frozen target-free graph: reciprocal bonuses `0`, endpoint rank `0.02`, ambiguity penalty `0.01`.

Do not alter these bytes and still call the result the verified v10 keeper.

## Frozen v10 quality evidence

| Candidate | Strict OOF | Graph OOF | Target-stress mean |
|---|---:|---:|---:|
| structured_only | `0.5808404006` | `0.5821464488` | `0.4355474106` |
| **no_teacher** | **`0.5931387077`** | **`0.5950413763`** | **`0.4496152683`** |
| no_contrastive | `0.5928725263` | `0.5978943607` | `0.4535367991` |

`no_contrastive` is locally strongest but retains the pair cross-encoder teacher and is not the production direction. The selected graph improves all five immutable folds. Target-stress is diagnostic and is not a leaderboard score.

## Runtime proof

Exact archive SHA, organizer image `odsai/ecup26-matching-baseline:1.0`, RTX 2060 SUPER:

| Gate | Rows | Acceptance | Outer inference wall | Headroom | Result |
|---|---:|---:|---:|---:|---|
| public-size | `115000` | `330 s` | **`173.842174445 s`** | **`156.157825555 s`** | **PASS** |
| private-size | `275000` | `700 s` | **`391.608035937 s`** | **`308.391964063 s`** | **PASS** |

Private keeper gate `MakSoS1/gpu-dispatch#31692817075`, artifact `9178292328`: return code `0`, valid exact pair order, finite/nonconstant numeric scores, `271964` unique scores, teacher checkpoint absent.

The old `<120/<250` thresholds were an exploratory over-strict tuning target and are not organizer limits or keeper acceptance.

Do not require graph-rescored ranking scores to be clipped to `[0,1]` unless the competition contract explicitly requires probabilities. v10 produced valid continuous ranking scores from `-0.0118716200922457` to `1.02` in the private gate.

## HF publication

Run `31693414226`: SUCCESS.

Verified private paths:

- `submissions/v10/final/ecup-v10-no-teacher-graph-0.5950413763-submission.zip`;
- `submissions/v10/final/V10_KEEPER.json`.

## Mandatory reading order

1. `ecup_matching/experiments/CURRENT.json`
2. `docs/agent-memory/PROJECT_STATE.md`
3. `docs/agent-memory/EXPERIMENT_INDEX.md`
4. `ecup_matching/experiments/v10/PLAN.md`
5. `ecup_matching/experiments/v10/RESULTS.md`
6. `ecup_matching/experiments/v10/SAFE_METRICS.json`
7. `docs/agent-memory/DECISIONS.md`
8. `docs/agent-memory/SECURITY.md`
9. `docs/agent-memory/ITERATION_PROTOCOL.md`

## Binding invariants

- Never change the split to improve a metric.
- Never inspect/use sealed-gold labels/items for tuning, mining, calibration or runtime decisions.
- Every target-fitted development layer requires outer cross-fitting.
- Production refit is never validation.
- Leaderboard, strict OOF and target-stress remain separate evidence axes.
- A single leaderboard score is not converted into row-level labels or fitted calibration.
- Outside-container wall is authoritative for timeout safety.
- A smoke test is compatibility evidence, not runtime evidence.
- Never restore pair-teacher inference just for its local metric gain without new exact end-to-end runtime evidence.
- Never weaken a real production gate after observing failure; distinguish experimental tuning targets from production acceptance before drawing conclusions.

## Persistent memory

Hardened Memora pin: `bc64ff745a9b2c0e6245e0137654f041fba0c155`. Graph, LLM, external embeddings and auto-capture remain disabled. Raw competition data, model weights, submission ZIPs and the memory DB remain private.

Current checkpoint target: `--iteration v10`.

## Next external action

Submit the **exact** v10 keeper archive from private HF. When the platform finishes, record the measured v10 leaderboard score as a separate external observation without rewriting local validation history.
