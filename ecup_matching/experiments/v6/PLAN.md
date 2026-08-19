# E-CUP Matching — Iteration v6 Runtime-Constrained Plan

Date: 2026-08-11
Status: in progress

## Goal

Produce a competition submission that simultaneously satisfies:

1. strict component-disjoint development OOF Macro AP `>= 0.6000`;
2. organizer-compatible offline inference;
3. end-to-end runtime low enough for the competition execution limit;
4. sealed gold remains unopened (`gold_rows_scored = 0`).

Quality above `0.6000` is a hard gate. Once the gate is crossed, runtime is the primary optimization objective.

## Immutable validation contract

- human pairs: `365,654`;
- development rows: `285,210`;
- sealed-gold rows: `80,444`;
- immutable development folds: `5`;
- cross-split item/component overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- metric: unweighted mean of `sklearn.metrics.average_precision_score` over exactly the 20 official categories;
- sealed gold must not be opened or scored during v6 architecture/runtime selection.

## Starting point

v5 quality-best:

- architecture: category-shrunk simplex + HGB equal-rank fusion over six signals;
- strict OOF Macro AP: `0.6018115534135564`;
- exact organizer smoke: passed;
- competition runtime: too slow for the desired production constraint.

The v6 task is therefore not to maximize local AP at any cost. It is to find a Pareto point with AP `>= 0.6000` and materially lower runtime.

## Candidate ladder

Evaluate in this order, keeping all meta predictions genuinely outer-cross-fitted:

1. remove pair teacher and refit meta;
2. remove contrastive signal and refit meta;
3. structured-only refit;
4. target-free partial-teacher gate based only on disagreement among cheap signals and category;
5. leakage-safe teacher distillation;
6. student + real-teacher hybrid;
7. if the minimum-quality candidate remains too slow, optimize the retained neural inference path without changing semantics before considering lower-precision or architecture changes.

No candidate is retained merely because it is faster. It must first pass strict OOF `>= 0.6000`.

## Selected quality candidate

The first retained runtime-constrained point is gate95:

- real pair teacher on the target-free top-disagreement `95%` of pairs inside each category;
- teacher surrogate on the other `5%` = mean percentile rank of the five non-teacher signals;
- retained non-teacher signals: weak, sparse, explicit, supervised contrastive, typed explicit;
- category-shrunk + fixed HGB 50/50 rank fusion refit under the same outer-fold protocol;
- strict OOF Macro AP: `0.6006003614522999`;
- actual development teacher fraction: `0.9500262964131693`.

This is deliberately a small quality margin over the hard `0.6000` threshold, so FP32 neural semantics are retained until a lower-precision path is separately quality-verified.

## Runtime implementation plan

For the retained gate95 candidate:

- length-bucket contrastive item texts before batching;
- length-bucket selected teacher pairs before batching;
- use RTX/VRAM-aware CUDA batch sizes;
- use non-blocking CUDA transfers;
- request SDPA when supported, with eager fallback;
- implement CUDA OOM batch-halving fallback;
- keep inference offline and local-files-only;
- emit phase timings for load, structured, contrastive, gate, teacher, meta and write.

Target runtime hardware for development profiling: self-hosted `ecup-rtx2060`, NVIDIA GeForce RTX 2060 SUPER, 8 GiB VRAM, using the exact organizer image and the exact candidate ZIP bytes.

## Required production gates

A v6 archive is not final until all of the following pass on the same candidate architecture:

1. selected v6 contract tests;
2. deterministic production meta refit on all development OOF evidence;
3. verified base package SHA before patching;
4. ZIP integrity and size check;
5. exact organizer-image offline/read-only smoke with correct `id1,id2,predict` schema and finite nonconstant scores;
6. full repository test suite and documentation policy;
7. SHA-256 provenance record;
8. exact-byte RTX 2060 benchmark, including a full reference `matches.parquet` run;
9. private Hugging Face copy and GitHub Actions artifact of the exact retained ZIP;
10. documentation updated with measured runtime and final artifact identifiers.

## Abort / architecture-change criteria

- Reject any architecture with strict OOF `< 0.6000`.
- Do not infer GPU runtime from CPU smoke or partial samples.
- If exact gate95 full-run runtime is still outside the execution budget, preserve the `0.6000` gate and continue optimization. Candidate next steps are larger safe FP32 batches, tokenizer/data-path optimization, then separately validated mixed precision or a different distilled architecture.
- Never use sealed-gold labels to recover a lost `0.6000` margin.

## Evidence separation

Keep these axes separate in all reports:

- strict local OOF quality;
- organizer compatibility/smoke status;
- measured GPU runtime;
- Public leaderboard score;
- Private leaderboard score.

A production refit score, CPU smoke duration, or future leaderboard result must never overwrite the strict OOF selection metric.
