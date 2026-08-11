# E-CUP Matching — Canonical Project State

Updated: 2026-08-12
Current iteration: **v6**

## Objective

Maximize E-CUP 2026 product-matching Macro AP while preserving honest unseen-product validation and producing an offline organizer-compatible submission that also fits the competition runtime constraint.

The v6 decision rule is Pareto-ordered:

1. strict component-disjoint OOF Macro AP must be `>= 0.6000`;
2. among candidates that pass quality, minimize measured end-to-end inference runtime;
3. keep sealed gold unopened.

## Current state — read this first

- **Best strict local quality reference:** v5, `0.6018115534135564` OOF Macro AP.
- **Current runtime-constrained iteration:** v6.
- **Selected v6 candidate:** `v6-fast-gate95-category-shrunk-hgb`.
- **Selected v6 strict OOF:** `0.6006003614522999`.
- Requested teacher coverage: `0.95`; actual development fraction: `0.9500262964131693`.
- v6 uses the real pair teacher only for the target-free highest-disagreement 95% of pairs per category and a five-signal rank surrogate elsewhere.
- Exact organizer-image offline/read-only 64-row smoke for the current code path: **passed** in run `31535674086`.
- That run reached `264 passed, 1 failed`; the only failure was stale documentation-memory policy, not inference/model behavior.
- The candidate ZIP from that pre-policy run was `1,143,630,143` bytes with SHA-256 `20c5f128e43c5303893301f012726381df06a4e20d027ea054acf36e0f6aae40`, but it is **not retained as final** because the full repository gate was not GREEN.
- Exact rebuilt v6 ZIP + RTX 2060 full-run benchmark: **pending**.
- Sealed gold: **unopened**, `0` rows scored.
- Public/private leaderboard score: **unknown**. Submission attempts so far were rejected on the platform time limit, not scored.
- Root cause of those timeouts is identified and fixed: the structured phase ran single-threaded and projected to `~254s` of the `360s` public budget and `~608s` of the `780s` private budget on its own. It is now `~22s` and `~44s`, with bitwise identical predictions.

Do not report the v5 or v6 local OOF number as a Public/Private leaderboard result.

## Immutable validation protocol

- human labels: `365,654` rows;
- connected item components: `345,654`;
- development rows: `285,210`;
- sealed-gold rows: `80,444`;
- five immutable development folds;
- cross-split item/component overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- strict metric: `average_precision_score` per official category, unweighted mean over exactly 20 categories.

Every target-fitted meta layer must be genuinely outer-cross-fitted. Full-development production refits are not validation. Sealed-gold labels are not available for architecture choice, runtime tuning, mining or calibration.

## v5 quality reference

v5 final quality-first architecture uses six signals:

1. weak category specialist;
2. sparse TF-IDF specialist;
3. explicit per-key attribute specialist;
4. supervised contrastive item score;
5. pair-teacher score;
6. typed/canonicalized explicit specialist.

It combines a fixed category-shrunk simplex and fixed HGB meta stack with a frozen 50/50 percentile-rank fusion.

Strict OOF: `0.6018115534135564`.

Verified v5 package:

- ZIP: `ecup-v5-category-hgb-fusion-0.6018115534-submission.zip`;
- SHA-256: `442769bd2c92d43730d7034fb91d8a83e596a8445ae3c3f887783890e90284d5`;
- private HF: `submissions/v5/0.6018115534`;
- Actions artifact `9116032675`;
- exact organizer smoke passed.

v5 remains the quality reference/fallback, but v6 is the active iteration because runtime is now a hard production dimension.

## v6 quality/runtime exploration

| Candidate | Strict OOF Macro AP | Decision |
|---|---:|---|
| structured only | `0.5808404005946962` | reject |
| no teacher | `0.5931387077244183` | reject |
| no contrastive | `0.5928725263319000` | reject |
| teacher gate 25% | `0.5929214688140778` | reject |
| teacher gate 55% | `0.5966896566149946` | reject |
| teacher gate 85% | `0.5999300791828578` | reject narrowly |
| distilled teacher | `0.5931935841654697` | reject |
| student + real teacher 85% | `0.5998746122650258` | reject narrowly |
| **teacher gate 95%** | **`0.6006003614522999`** | **current v6 candidate** |

The 95% gate is intentionally the lowest retained quality point found so far. Lower teacher coverage did not clear the hard `0.6000` quality gate under honest outer OOF.

## Current v6 architecture

The five non-teacher signals remain weak, sparse, explicit, supervised contrastive and typed explicit. They are converted to target-free percentile ranks and their disagreement is measured per pair. Inside each official category, the real teacher is evaluated for the highest-disagreement 95% of pairs.

For the other 5%, the teacher signal is replaced by the unweighted mean of the five non-teacher percentile ranks. Selected teacher outputs are percentile-ranked only among selected teacher rows. The final category-shrunk + HGB meta stack is trained/evaluated with the same gated signal under outer cross-fitting.

Selection evidence:

- run `31531141700`, job `93911179929`;
- source `fb15ec43a90c892c416acb2d10fe04cc126a4398`;
- private HF `experiments/v6/teacher-gate/95/fb15ec43a90c`.

## v6 runtime implementation

The quality margin above `0.6000` is small, so current production inference deliberately retains FP32 neural semantics. Speedups are semantic-preserving implementation changes:

- stable length bucketing for contrastive item texts;
- stable length bucketing for selected teacher pairs;
- 8 GiB RTX defaults: contrastive batch `256`, teacher batch `96`;
- larger batches on larger VRAM;
- CUDA OOM batch-halving fallback;
- non-blocking CUDA transfers;
- SDPA requested where supported, eager fallback otherwise;
- structured feature chunking pinned at `10,000`, scored across `fork` worker processes at unchanged chunk boundaries;
- `difflib` ratios shared between the legacy and typed structured passes;
- one shared `ItemNorm` pass behind the contrastive and teacher text caches;
- single-pass `select_items_by_ids`;
- CUDA probed only after the structured pool is done, so no worker inherits a CUDA context;
- offline/local-files-only inference;
- phase telemetry for load, structured, contrastive, gate, teacher, meta and write.

The authoritative runtime benchmark must execute the **exact final ZIP bytes** on self-hosted `ecup-rtx2060` inside `odsai/ecup26-matching-baseline:1.0` with `--gpus all`, including a full reference `matches.parquet` run. CPU smoke timing is compatibility evidence, not the production runtime claim.

## Current production-gate evidence

Run `31535674086`, source `4da50f66942472b8e8b70270cbeb00639930b6b5`:

- selected v6 contract tests: passed;
- production category/HGB refit: passed;
- verified v5 six-signal base SHA: passed;
- current runtime package creation: passed;
- exact organizer-image offline/read-only smoke: passed;
- 64-row CPU smoke phases: load `6.696s`, structured `0.913s`, contrastive `7.864s`, gate `0.002s`, teacher `8.654s`, meta `0.008s`, write `0.003s`, total `24.14s`;
- output schema/order/finite/nonconstant checks: passed;
- full suite: `264 passed, 1 failed, 91 warnings`;
- sole failure: documentation memory-policy state (`CURRENT.status` and experiment index formatting).

Because the repository gate was not GREEN, that archive was not uploaded as final and the temporary GPU-benchmark release was not created. The documentation state is now being corrected before rebuilding.

## Binding failure lessons

- Infrastructure, OOM, packaging or API failures are not model scores.
- Do not weaken full tests to publish an artifact.
- Production refit scores are not validation.
- Do not use sealed gold to recover a runtime-induced quality loss.
- Direct attribute score shifts failed while explicit per-key estimator features were useful; do not conflate them.
- Pretrained-only embeddings were weak; supervised contrastive was the useful neural signal.
- Any mixed-precision/quantized path that can alter ordering requires its own honest quality verification before replacing the FP32 candidate.
- A fixed-overhead smoke is not runtime evidence. The 64-row CPU smoke could not reveal a per-pair cost problem; only a run within an order of magnitude of the private pair count can.
- The submission file list must be derived from the import graph, never hand-maintained: a stale module copied from the base archive changes predictions silently.
- Structured chunk size is not a free parameter; float32 GEMM batching makes it perturb scores.

## Current files to read

1. `ecup_matching/experiments/CURRENT.json`
2. `ecup_matching/experiments/v6/PLAN.md`
3. `ecup_matching/experiments/v6/RESULTS.md`
4. `ecup_matching/experiments/v6/SAFE_METRICS.json`
5. `docs/agent-memory/EXPERIMENT_INDEX.md`
6. `docs/agent-memory/DECISIONS.md`
7. `docs/agent-memory/SECURITY.md`
8. `docs/agent-memory/ITERATION_PROTOCOL.md`

## Next action

Make the repository policy GREEN, rebuild the exact v6 gate95 ZIP, publish immutable SHA/provenance, run the exact same bytes on `ecup-rtx2060` in the organizer image including a full reference run, and retain the archive only if the measured runtime fits the execution budget. If not, continue architecture/runtime optimization while preserving strict OOF `>= 0.6000` and unopened sealed gold.
