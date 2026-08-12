# v9 Results — leaderboard-adapted runtime-safe submission

Status: **IN PROGRESS — exact keeper runtime gate running**

Updated: 2026-08-12

## Why v9 exists

The owner reports that v7 is currently the best successfully scored submission at approximately `0.36` leaderboard AP, while v8 again returned `Container did not finish in time`. The v7 platform score is stored only as an external anchor and is not used to fit v9.

The decisive v8 runtime bug is now identified: exact gate70 evidence showed inner `run.py` around `731.22 s` but true outside-container wall `820.784 s`. The old workflow still wrote `full_runtime_gate_passed=true` because it accepted the process exit/output without comparing the measured outer wall against the intended budget. v9 closes that failure mode.

## Validation v2

Frozen immutable split:

- development rows `285,210`;
- sealed gold rows `80,444`;
- five component-disjoint folds;
- cross-split item overlap `0`;
- split SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- official metric = unweighted Macro AP over exactly 20 categories;
- sealed gold opened `false`, rows scored `0`.

Validation run: `31639183423`.

Frozen target-stress prevalence ratio: `0.566880890615799`.

Frozen graph config: reciprocal-best `0`, reciprocal-top3 `0`, endpoint-rank `0.02`, ambiguity penalty `0.01`.

| Candidate | Teacher fraction | Strict OOF | Fold-local graph OOF | Graph delta | Target-stress mean | Stress std | p05–p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate25 | `0.2500227902` | `0.5947115591` | `0.5961903713` | `+0.0014788122` | `0.4507779206` | `0.0016871299` | `0.4481787745–0.4534007374` |
| **gate40** | **`0.4000245433`** | **`0.5955054274`** | **`0.5970059311`** | **`+0.0015005037`** | **`0.4515676235`** | **`0.0017001454`** | **`0.4489228671–0.4542054080`** |

All five folds have positive graph delta for both candidates. Gate40 was selected **before** the final runtime measurement because it dominates gate25 on strict OOF, graph OOF and target-stress mean. No teacher coverage was invented after seeing this result: 25/40/55/70/85/95 were already frozen candidates.

The owner-reported v7 leaderboard value `~0.36` has `used_for_fitting=false`. The desired v9 leaderboard region `~0.5` is a target only. Measured v9 leaderboard score is `null` until the platform actually scores the archive.

## Gate40 production refit

Run `31639692541` completed successfully:

- targeted tests: `15 passed`;
- all development rows used: `285,210`;
- actual real-teacher fraction: approximately `0.400025`;
- elapsed: `74.4 s`;
- peak RAM: `0.736 GiB`;
- Actions artifact `9158411928`;
- private HF prefix `experiments/v9/production/gate40/853a3925ac2b`;
- sealed gold remained unopened.

Production refit is not validation; the selection score remains the frozen outer-OOF evidence above.

## Exact gate40 package

Final build run `31640050373` is GREEN.

Exact candidate:

- file: `ecup-v9-gate40-fp16-graph-0.5970059311-submission.zip`;
- bytes: `1,251,659,961`;
- SHA-256: `925456cde1e47c50dc0141ce64bed5ef00d9f574152f285869ebea2db6935782`;
- release tag: `ecup-v9-gate40-final-eb2bcf18d53e`;
- local reassembly from five no-recompression transfer parts reproduced the same byte count and SHA.

Build verification checked the complete optimized import closure, stale runtime exclusion, package metadata, path safety, duplicate members, required model/runtime files, graph config and sealed-gold fields.

Packaged runtime policy:

- structured worker cap `8`;
- CUDA FP16/autocast;
- contrastive batch `256` on RTX 2060;
- teacher batch `96` on RTX 2060;
- outer-wall acceptance `700.0 s`;
- watchdog `720 s`.

## Corrected exact runtime gate

Private run `MakSoS1/gpu-dispatch#31640233511` is the authoritative gate.

It uses:

- the exact package SHA above;
- organizer image;
- RTX 2060 SUPER;
- exact `275,000` pair fixture;
- outside-container wall measurement;
- output schema/order/finite/nonconstant validation;
- acceptance only if `return_code=0`, output valid and `wall_seconds <=700.0`;
- independent watchdog at `720 s`.

At the time this document was written, the organizer-image inference step was still running. Therefore **no final runtime pass is claimed yet**.

## Predeclared gate25 fallback

Gate25 production refit has already completed in run `31640425364`, artifact `9158679674`, so a runtime veto does not require a new research loop.

Frozen fallback quality evidence:

- strict OOF `0.5947115591000889`;
- fold-local graph OOF `0.5961903713277379`;
- target-stress mean `0.45077792061326727`;
- teacher coverage `0.25`;
- sealed gold unopened.

If gate40 exceeds `700 s`, the threshold is **not** relaxed. The final v9 package is rebuilt with gate25 and subjected to the same exact outer-wall gate.

## KEEP / REJECT / FAIL log

- **REJECT v8 as submission:** platform timed out again; old runtime pass criterion was invalid.
- **KEEP outer-wall gate:** `wall_seconds` is now authoritative, not inner telemetry/exit status.
- **KEEP target-stress as diagnostic:** it captures a documented retrieval/human prevalence shift but does not replace strict OOF.
- **KEEP graph config:** fold-local graph delta is positive on all five folds for gate25 and gate40.
- **KEEP gate40 before runtime:** dominates gate25 on strict OOF + graph OOF + stress.
- **KEEP gate25 as fallback:** predeclared, lower-cost, production-refit already complete.
- **DO NOT CLAIM `~0.5`:** it is the desired leaderboard region, not measured evidence.
- **DO NOT OPEN GOLD:** sealed gold remains unopened/0 rows scored.

## Remaining completion steps

1. Finish corrected exact gate40 runtime run `31640233511`.
2. If it fails the frozen `<=700 s` rule, activate gate25 and repeat the exact package/runtime gate.
3. Freeze final package SHA and runtime evidence in this file and `SAFE_METRICS.json`.
4. Mark `CURRENT.json` completed only for the exact keeper.
5. Run full repository tests and `scripts/memory_policy.py`.
6. Run hardened Memora ingest and private HF checkpoint for `v9`.
7. Hand off only the byte-verified keeper ZIP.
