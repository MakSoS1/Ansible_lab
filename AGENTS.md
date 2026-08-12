# AGENTS.md — READ THIS FIRST

Mandatory entry point for agents working on E-CUP matching. Do not train, tune, inspect sealed-gold labels, change validation semantics, or publish a submission before reading the canonical state below.

## Project objective

Solve ODS E-CUP 2026 product matching with honest unseen-product validation and a submission that finishes within the organizer runtime limit. The strict local metric is unweighted Macro Average Precision over exactly 20 official categories. Local OOF, target-stress diagnostics and platform leaderboard scores are different evidence axes and must never be conflated.

## Current iteration: v9

- Working branch: `ecup-v9-leaderboard-adapt`; do not modify `main` for v9 work unless explicitly requested.
- Private artifact repo: `Maksim123321/e-cup-2026-matching-private`.
- Home GPU is isolated through private `MakSoS1/gpu-dispatch`.
- GPU target: `ecup-rtx2060`, NVIDIA GeForce RTX 2060 SUPER, 8 GiB VRAM.
- Immutable split: `285,210` development rows + `80,444` sealed-gold rows, five component-disjoint folds, cross-split item overlap `0`.
- Split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed gold remains unopened: `gold_metric_opened=false`, `gold_rows_scored=0`.
- Owner-reported successfully scored v7 leaderboard anchor: approximately `0.36`. It is external evidence only and `used_for_fitting=false`.
- v9 desired leaderboard region: approximately `0.5`; this is a target, **not a claimed score**.
- v8 is closed as a runtime failure: inner `run.py` around `731.22 s`, true outer wall `820.784 s`, and the old workflow incorrectly called it a pass because it did not enforce the measured outer wall.
- Current v9 candidate before final runtime: gate40 + FP16 + structured cap8 + fold-local target-free graph.
- Predeclared fallback: gate25; use it only if gate40 fails the frozen exact runtime gate. Never relax the runtime gate after seeing a result.

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
11. `ecup_matching/SOLUTION_RESEARCH.md`
12. `ecup_matching/BASELINE_CONTRACT.md`

Historical v5/v6/v7/v8 documents remain evidence and failure lessons but do not redefine the current v9 selection/runtime contract.

## Validation invariants

- Never alter split SHA `aae58f...eb55b` to improve a metric.
- Never inspect/use sealed-gold labels during development, target-stress construction, graph tuning, runtime tuning or calibration.
- Do not encode/mine sealed-gold items as adaptation or weak-label data.
- Every target-fitted model or stack must score each held component using parameters fit without labels from that component.
- Production refit on all development labels is allowed only after selection and is never a validation score.
- Platform leaderboard results remain separate from strict local OOF.
- A single leaderboard score must not be converted into row-level labels, direct probability calibration, or a fitted mapping without additional legitimate evidence.
- Target-stress is diagnostic only; it never replaces official strict OOF.

## v9 validation v2

Frozen source run: `31639183423`.

| Candidate | Teacher fraction | Strict OOF | Fold-local graph OOF | Target-stress mean |
|---|---:|---:|---:|---:|
| gate25 | `0.2500227902` | `0.5947115591` | `0.5961903713` | `0.4507779206` |
| **gate40** | **`0.4000245433`** | **`0.5955054274`** | **`0.5970059311`** | **`0.4515676235`** |

Both candidates improve with the frozen graph config on all five folds. Gate40 was selected before final runtime because it dominates gate25 on strict OOF, graph OOF and target stress. Gate25 is the predeclared fallback.

Frozen target-stress prevalence ratio: `0.566880890615799`.

Frozen graph config:

- reciprocal-best bonus `0`;
- reciprocal-top3 bonus `0`;
- endpoint-rank weight `0.02`;
- ambiguity penalty `0.01`.

The v7 leaderboard anchor `~0.36` did not enter fitting.

## Current v9 package

Selected gate40 production refit:

- run `31639692541`;
- artifact `9158411928`;
- full development rows `285,210`;
- actual teacher fraction `~0.400025`;
- elapsed `74.4 s`;
- peak RAM `0.736 GiB`;
- private HF `experiments/v9/production/gate40/853a3925ac2b`.

Exact candidate package:

- `ecup-v9-gate40-fp16-graph-0.5970059311-submission.zip`;
- `1,251,659,961` bytes;
- SHA-256 `925456cde1e47c50dc0141ce64bed5ef00d9f574152f285869ebea2db6935782`;
- build run `31640050373`;
- release tag `ecup-v9-gate40-final-eb2bcf18d53e`.

Runtime semantics in the package:

- complete first-party import closure;
- structured worker cap `8`;
- CUDA FP16/autocast;
- contrastive batch `256` on RTX 2060;
- teacher batch `96` on RTX 2060;
- frozen graph rescore above.

## Exact runtime gate — binding

Corrected gate40 run: private `MakSoS1/gpu-dispatch` run `31640233511`.

The candidate passes only if all conditions hold on the exact final ZIP bytes in organizer image over exactly `275,000` pairs:

1. exit code `0`;
2. output row count/schema/pair order valid;
3. scores finite, bounded and nonconstant;
4. **outer wall `<=700.0 s`**;
5. watchdog `720 s`.

`700.001 s` fails. A zero exit code never overrides an outer-wall failure. Never revert to the old v8 criterion.

If gate40 fails, activate gate25. Gate25 production refit already exists from run `31640425364`, artifact `9158679674`; repeat package verification and the same exact runtime gate. Do not weaken the threshold.

## Retained runtime rules

- score structured chunks across a fork worker pool at unchanged chunk boundaries;
- structured worker count is capped at `8` from measured sweep;
- share duplicated normalization/difflib work where semantics are identical;
- single-pass item selection;
- defer CUDA initialization until CPU fork work is done;
- stable length bucketing;
- VRAM-aware batches and OOM fallback;
- CUDA FP16 only on the validated path;
- offline/local-files-only inference;
- phase telemetry for load, structured, neural, meta/graph and write;
- submission file list comes from `ecup_matching/ci/runtime_closure.py`, never a manual copy list.

A smoke test is compatibility evidence only. Runtime success requires the exact packaged path at near-private scale, with outside-container wall measurement.

## Binding failure lessons

- Infrastructure/OOM/API failures are not model-quality evidence.
- Production refit scores are not validation.
- High single-fold research diagnostics are not strict OOF.
- v7 fold-0 `~0.70` did not imply comparable platform performance.
- Never use sealed gold to recover a leaderboard or runtime gap.
- Never tune directly to one leaderboard score.
- Mixed precision/quantization requires quality/ranking verification.
- Outer wall is authoritative for timeout safety.
- Never weaken a test/runtime gate to publish a package.

## v9 completion gate

v9 is completed only when the same keeper architecture has:

1. frozen validation provenance recorded;
2. production refit reproducible;
3. package integrity/CRC/import closure verified;
4. exact `275k` organizer-image RTX outer wall `<=700 s` with valid output;
5. sealed gold still unopened;
6. final package SHA/size fixed;
7. full repository tests GREEN;
8. `scripts/memory_policy.py` GREEN;
9. hardened Memora ingest and private HF checkpoint GREEN;
10. final archive handed off. Actual leaderboard AP remains unknown until the platform scores it.

## Persistent memory / security

Hardened Memora pin: `bc64ff745a9b2c0e6245e0137654f041fba0c155`. Local SQLite + TF-IDF only; graph/LLM/external embeddings/auto-capture disabled. Raw competition data, model weights, submission ZIPs and memory DB remain private.

Machine-readable memory sources include `ecup_matching/experiments/CURRENT.json` and `ecup_matching/experiments/v*/SAFE_METRICS.json`.

## Iteration / handoff protocol

After every meaningful KEEP/REJECT/FAIL:

1. update `v9/RESULTS.md`;
2. update `v9/SAFE_METRICS.json`;
3. update `CURRENT.json` when keeper/status/next action changes;
4. update `EXPERIMENT_INDEX.md` and `PROJECT_STATE.md`;
5. preserve durable lessons;
6. only from GREEN repository state, run full tests, memory policy, hardened Memora ingest and checkpoint.

Current checkpoint target is `--iteration v9`.
