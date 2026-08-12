# E-CUP Matching — Canonical Project State

Updated: 2026-08-12
Current iteration: **v9**

## Objective

Maximize E-CUP 2026 product-matching Macro AP while preserving honest unseen-product validation and producing an offline organizer-compatible submission that finishes comfortably inside the platform runtime limit.

The platform has now supplied one crucial external anchor: the owner reports that v7 successfully scored at approximately **0.36** on the leaderboard. That value is an external outcome only. It is not a training label, is not used to fit/calibrate v9, and must never overwrite local OOF. The v9 goal is to move toward the `~0.5` leaderboard region, but no v9 leaderboard score may be claimed until the platform actually scores it.

## Current state — v9

v8 is closed as a runtime failure. The corrected diagnosis is that the old gate declared success from process exit/output even though the true outside-container wall exceeded the intended budget. On exact gate70 evidence, `run.py` completed in roughly `731.22 s`, while measured outer wall was `820.784 s`; the platform subsequently timed out again. v9 makes outer wall authoritative and gives runtime veto power over architecture selection.

Current selected candidate before final runtime is **gate40**:

- architecture: retained v5/v6 six-signal family with selective real pair teacher at 40% target-free disagreement coverage;
- corrected complete runtime import closure;
- structured worker cap `8`;
- CUDA FP16/autocast;
- RTX batches `contrastive=256`, `teacher=96`;
- fold-local target-free graph postprocess with frozen config `rb=0`, `rt=0`, `ep=0.02`, `ap=0.01`;
- exact package: `ecup-v9-gate40-fp16-graph-0.5970059311-submission.zip`;
- package bytes: `1,251,659,961`;
- package SHA-256: `925456cde1e47c50dc0141ce64bed5ef00d9f574152f285869ebea2db6935782`;
- final build run: `31640050373`;
- prerelease tag: `ecup-v9-gate40-final-eb2bcf18d53e`.

A fully refit **gate25** fallback is already prepared. It is activated only if gate40 fails the frozen exact runtime gate. The runtime threshold is never weakened after seeing a result.

## Immutable validation protocol

- human labels: `365,654` rows;
- connected item components: `345,654`;
- development rows: `285,210`;
- sealed-gold rows: `80,444`;
- five immutable development folds;
- cross-split item/component overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- strict metric: `average_precision_score` per official category, unweighted mean over exactly 20 categories;
- sealed gold remains unopened and `0` rows have been scored.

Every target-fitted layer must be genuinely outer-cross-fitted. Full-development production refits are not validation. Sealed-gold labels/items are unavailable for architecture choice, runtime tuning, mining or calibration.

## Validation v2 and leaderboard-shift evidence

v9 keeps strict OOF and adds a separate target-stress diagnostic based on the previously measured retrieval/human prevalence ratio `0.566880890615799`. This diagnostic does not replace the official metric and does not inspect sealed gold.

| Candidate | Teacher fraction | Strict OOF | Fold-local graph OOF | Target-stress mean | Graph positive on all folds |
|---|---:|---:|---:|---:|---|
| gate25 | `0.2500227902` | `0.5947115591` | `0.5961903713` | `0.4507779206` | yes |
| **gate40** | **`0.4000245433`** | **`0.5955054274`** | **`0.5970059311`** | **`0.4515676235`** | **yes** |

Run `31639183423` produced this comparison. Gate40 was selected before the final runtime measurement because it dominates gate25 on strict OOF, graph OOF and target stress while materially reducing pair-teacher work relative to gate70. Gate25 remains the predeclared runtime fallback.

The owner-reported v7 leaderboard score `~0.36` is stored as `used_for_fitting=false`. The v9 target region near `0.5` is not a measured score.

## Production refit

Selected gate40 was refit on all `285,210` development rows in run `31639692541`:

- tests: `15` targeted tests passed in the refit gate;
- actual teacher fraction: approximately `0.400025`;
- refit time: `74.4 s`;
- peak RAM: `0.736 GiB`;
- Actions artifact: `9158411928`;
- private HF prefix: `experiments/v9/production/gate40/853a3925ac2b`;
- sealed gold remained unopened.

The production refit score itself is never reported as validation; selection evidence is the frozen outer-OOF table above.

## Runtime completion contract

The old v8 runtime criterion is invalid. v9 uses the exact final ZIP bytes in organizer image on the RTX 2060 SUPER and an exact `275,000`-pair fixture.

A package passes only when all are true:

1. outer wall is measured from immediately before organizer-container launch until container exit;
2. exit code is zero;
3. output schema/order/row count are correct and scores are finite/nonconstant;
4. `wall_seconds <= 700.0`;
5. independent watchdog is `720 s`.

The acceptance limit is intentionally stricter than the nominal private platform limit to preserve margin. `700.001 s` fails. A zero process exit cannot override an outer-wall failure.

Corrected private GPU gate run for gate40: `MakSoS1/gpu-dispatch` run `31640233511`. Until it completes, v9 remains `in_progress`.

## Gate25 fallback

The fallback is not a post-result invention: 25% was one of the original frozen teacher coverages. Its production refit already completed successfully in public run `31640425364`, artifact `9158679674`, with validation fixed at:

- strict OOF `0.5947115591000889`;
- fold-local graph OOF `0.5961903713277379`;
- target-stress mean `0.45077792061326727`;
- sealed gold unopened.

If gate40 exceeds the 700-second acceptance ceiling, v9 moves to gate25 and repeats the exact package + outer-wall gate. The threshold is not relaxed.

## Historical quality/runtime references

- v5 quality reference: strict OOF `0.6018115534135564`;
- v6 gate95 reference: strict OOF `0.6006003614522999`;
- v7 diagnostic fold-0 values above `0.70` did not predict the observed platform score and were never honest five-fold strict OOF;
- v8 gate70 was invalid as a submission because true outer runtime exceeded the intended budget despite a misleading workflow pass marker.

This is why v9 optimizes a three-axis problem: honest strict OOF, target-distribution stress, and exact outer-wall runtime.

## Binding failure lessons

- Infrastructure, OOM, packaging or API failures are not model scores.
- Production refit scores are not validation.
- Public/private leaderboard evidence is separate from local OOF.
- Never use the leaderboard as row-level training data or fit a calibration from a single observed submission score.
- Never open sealed gold to recover a leaderboard/runtime gap.
- A fixed-overhead smoke is not runtime evidence.
- **Outer wall, not inner process time or exit code, is authoritative for timeout safety.**
- The final submission file list must be derived from the import graph, never hand-maintained.
- Structured chunk size remains pinned where float32 batching can perturb scores.
- Mixed precision is retained only with explicit quality/ranking evidence and exact package verification.
- Do not weaken runtime/quality tests to publish an artifact.

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

Finish the exact gate40 outer-wall run `31640233511`. If it passes `<=700 s`, freeze those exact ZIP bytes as v9. If it fails, activate the already-built gate25 fallback without changing the threshold. Then finalize v9 results/safe metrics, run repository tests + memory policy, ingest the canonical v9 state into hardened Memora, checkpoint to private HF, and hand off only the byte-verified keeper archive.
