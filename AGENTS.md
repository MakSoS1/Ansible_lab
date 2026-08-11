# AGENTS.md — READ THIS FIRST

Mandatory entry point for agents working on E-CUP matching. Do not train, tune, open validation labels or publish a submission before reading the canonical state files below.

## Project objective

Solve ODS E-CUP 2026 product matching with honest unseen-product validation. Official/local strict metric is unweighted Macro Average Precision over exactly 20 official categories. Item/component leakage, repeatedly tuned sealed holdouts and mixing local OOF with leaderboard evidence are prohibited.

## Current iteration: v7

- Working quality-sprint branch: `ecup-v7-neural`; do not modify/merge `main` unless explicitly requested.
- Retained runtime branch: `ecup-v6-fast-runtime`; retained publishing branch: `ecup-v6-fast`.
- Private artifact repo: `Maksim123321/e-cup-2026-matching-private`.
- Home GPU access is isolated through private `MakSoS1/gpu-dispatch`; public source never owns the self-hosted runner.
- GPU target: `ecup-rtx2060`, NVIDIA GeForce RTX 2060 SUPER, 8 GiB VRAM.
- Immutable split: `285,210` development rows + `80,444` sealed-gold rows, five component-disjoint folds, cross-split item overlap `0`.
- Split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed gold remains unopened: `gold_metric_opened=false`, `gold_rows_scored=0`.
- v5 remains the strict quality reference at `0.6018115534135564`.
- v6 gate95 remains the runtime reference at strict OOF `0.6006003614522999`; its prediction-preserving structured path is measured at `487.0 us/pair` in the local profile, but the exact full GPU runtime gate is still pending.
- v7 stretch target is strict OOF `0.70`. This is a target, not a score that may be claimed without five complete held-fold predictions.
- v7 Candidate A is an identity-first `ai-forever/ruBert-base` pair teacher with 256-token pair context, canonical typed attributes before residual numeric noise, and a leakage-safe weak+human curriculum with materially more training exposure than retained teacher2.
- Retained teacher2 already used ruBERT-base; the v7 hypothesis is context allocation + curriculum + training exposure, not merely swapping the model name.
- v5/v6 production semantics are not changed by v7 experiments until a v7 candidate wins strict OOF and runtime gates.

## Mandatory reading order

1. `ecup_matching/experiments/CURRENT.json`
2. `docs/agent-memory/PROJECT_STATE.md`
3. `docs/agent-memory/EXPERIMENT_INDEX.md`
4. `docs/agent-memory/DECISIONS.md`
5. `ecup_matching/experiments/v7/PLAN.md`
6. `ecup_matching/experiments/v7/RESULTS.md`
7. `ecup_matching/experiments/v7/SAFE_METRICS.json`
8. `ecup_matching/experiments/v6/PLAN.md`
9. `ecup_matching/experiments/v6/RESULTS.md`
10. `ecup_matching/experiments/v6/SAFE_METRICS.json`
11. `docs/agent-memory/SECURITY.md`
12. `docs/agent-memory/ITERATION_PROTOCOL.md`
13. `ecup_matching/SOLUTION_RESEARCH.md`
14. `ecup_matching/BASELINE_CONTRACT.md`

Historical plans/results remain useful for evidence and failure lessons but do not redefine the current v7 validation contract.

## Validation invariants

- Never alter split SHA `aae58f...eb55b` to improve a metric.
- Never inspect/use sealed-gold labels during development or runtime tuning.
- Do not encode/mine sealed-gold items as adaptation, hard-negative or weak-label data.
- Every target-fitted model or stack must score held components using parameters fit without those held labels.
- Production refit on all development labels is allowed only after selection and must never be reported as validation.
- `0.70` means honest strict development OOF if achieved, not a repeatedly tuned holdout score.
- Public/Private leaderboard scores are a separate evidence axis and must not overwrite local OOF.
- A shared weak pretraining checkpoint may be reused across outer folds only when the weak corpus excludes the complete human-item universe used by development/sealed-gold splitting; otherwise weak training must remain fold-safe.

## v7 architecture under test

Candidate A keeps v5/v6 structured and ensemble evidence as the comparison base but targets the underpowered pairwise signal directly:

1. `[NAME]`, `[BRAND]`, normalized `[MODEL]`/SKU first;
2. canonical typed identity attributes next (`storage_bytes`, `battery_mah`, diagonal, power, frequency, voltage, dimensions, mass/volume/count and other identity-bearing keys);
3. residual numeric evidence and low-priority attributes only after the identity packet;
4. ruBERT-base pair cross-encoder at `max_length=256`;
5. leakage-safe confidence-weighted weak curriculum plus authoritative human fine-tuning;
6. five immutable outer folds; only held-fold predictions form strict OOF;
7. no post-result fusion tuning on the same held labels without another nested layer.

Candidate B is allowed only if A is insufficient: an aligned shared-key pair view (`key: A || B`) evaluated under the same five-fold contract. It is not automatically retained.

## Retained v6 runtime rules

Runtime improvements already proven prediction-preserving remain binding infrastructure for any final candidate where applicable:

- score structured chunks across a `fork` worker pool at unchanged chunk boundaries;
- share `difflib` ratio results between legacy and typed structured passes;
- share one `ItemNorm` pass between contrastive and teacher text caches where serialization permits;
- single-pass `select_items_by_ids`;
- probe CUDA only after the structured pool has finished;
- stable length bucketing;
- VRAM-aware CUDA batches and CUDA OOM batch-halving fallback;
- non-blocking transfers;
- SDPA where supported with eager fallback;
- offline/local-files-only inference;
- phase telemetry for load, structured, neural stages, meta and write.

`STRUCTURED_CHUNK_SIZE` stays pinned at `10_000`. Float32 GEMM batching can perturb scores, so parallelism distributes existing chunks and never re-chunks.

A fixed-overhead smoke is compatibility evidence, not runtime evidence. Any final runtime claim needs a full reference `matches.parquet` run on the exact packaged path. Organizer budgets are 360 s public and 780 s private; retain safety margin.

The submission file list is derived from the import graph by `ecup_matching/ci/runtime_closure.py`. Never hand-maintain it.

## Retained / rejected evidence

KEEP:

- v5 quality reference: `0.6018115534135564`;
- v6 teacher gate 95% runtime reference: `0.6006003614522999`;
- v6 prediction-preserving structured optimization: `2210.1 -> 487.0 us/pair`, bitwise identical in its measured contract.

REJECT under the v6 quality gate:

- structured only `0.5808404005946962`;
- no teacher `0.5931387077244183`;
- no contrastive `0.5928725263319000`;
- teacher gate 25% `0.5929214688140778`;
- teacher gate 55% `0.5966896566149946`;
- teacher gate 85% `0.5999300791828578`;
- teacher distillation `0.5931935841654697`;
- student + real teacher 85% `0.5998746122650258`.

Infrastructure, OOM, runner, packaging and API failures are not model-quality evidence.

## v7 completion gate

A v7 submission is not completed until the same selected architecture has:

1. all targeted and full repository tests GREEN;
2. five complete immutable held-fold predictions and strict Macro AP over exactly 20 categories;
3. strict OOF above the retained v5 quality reference; `0.70` remains the stretch target;
4. sealed gold still unopened during selection;
5. deterministic production refit and verified source/base hashes;
6. exact organizer-image offline/read-only smoke;
7. final ZIP SHA/provenance and complete import closure;
8. exact production-path runtime benchmark with telemetry and a full reference `matches.parquet` run; the RTX 2060 is useful for feasibility/profiling, while final organizer feasibility must account for H100 execution;
9. private artifact persistence and documentation of KEEP/REJECT evidence;
10. `scripts/memory_policy.py`, Memora ingest and checkpoint GREEN.

Never weaken a test gate to publish a package and never invent the target metric.

## Persistent memory / security

Hardened Memora pin: `bc64ff745a9b2c0e6245e0137654f041fba0c155`. Local SQLite + TF-IDF only; graph/LLM/external embeddings/auto-capture disabled. Competition data, model weights and submission artifacts remain private. Public source contains reproducible code and source-backed handoff documentation, not credentials or private datasets.

Machine-readable memory sources must include:

- `ecup_matching/experiments/CURRENT.json`;
- `ecup_matching/experiments/v*/SAFE_METRICS.json`.

## Iteration / handoff protocol

Before implementation:

1. read the mandatory state sources;
2. preserve immutable split/gold rules;
3. update the current PLAN when hypothesis/runtime gates change;
4. use TDD/systematic debugging and distinguish RED-by-design from regressions.

After every meaningful KEEP/REJECT/FAIL:

1. update current `RESULTS.md`;
2. update current `SAFE_METRICS.json`;
3. update `CURRENT.json` when best/status/next action changes;
4. update `EXPERIMENT_INDEX.md` and `PROJECT_STATE.md`;
5. record durable lessons in `DECISIONS.md`;
6. only after the repository is GREEN, run full tests, memory policy, ingest and checkpoint for the current iteration.

Current checkpoint target is `--iteration v7`.