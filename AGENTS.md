# AGENTS.md — READ THIS FIRST

Mandatory entry point for agents working on E-CUP matching. Do not train, tune, open validation labels or publish a submission before reading the canonical state files below.

## Project objective

Solve ODS E-CUP 2026 product matching with honest unseen-product validation. Official/local strict metric is unweighted Macro Average Precision over exactly 20 official categories. Item/component leakage, repeatedly tuned sealed holdouts and mixing local OOF with leaderboard evidence are prohibited.

## Current iteration: v6

- Working branch: `ecup-v6-fast`; do not modify/merge `main` unless explicitly requested.
- Private artifact repo: `Maksim123321/e-cup-2026-matching-private`.
- Home GPU access is isolated through private `MakSoS1/gpu-dispatch`; public source never owns the self-hosted runner.
- GPU target: `ecup-rtx2060`, NVIDIA GeForce RTX 2060 SUPER, 8 GiB VRAM.
- Immutable split: `285,210` development rows + `80,444` sealed-gold rows, five component-disjoint folds, cross-split item overlap `0`.
- Split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed gold remains unopened: `gold_metric_opened=false`, `gold_rows_scored=0`.
- v5 remains the strict quality reference at `0.6018115534135564`.
- v6 hard rule: first require strict OOF `>= 0.6000`, then minimize measured end-to-end inference runtime.
- Current v6 candidate: target-free 95% pair-teacher disagreement gate + category-shrunk/HGB meta.
- Current v6 strict OOF: `0.6006003614522999`.
- Actual development teacher fraction: `0.9500262964131693`.
- Lower-cost candidates tested so far did not clear `0.6000`; do not silently replace gate95 with them.
- Exact organizer-image 64-row offline/read-only smoke for the current runtime path has passed.
- Exact rebuilt final ZIP and full RTX 2060 runtime gate are still required before v6 can be marked completed.

## Mandatory reading order

1. `ecup_matching/experiments/CURRENT.json`
2. `docs/agent-memory/PROJECT_STATE.md`
3. `docs/agent-memory/EXPERIMENT_INDEX.md`
4. `docs/agent-memory/DECISIONS.md`
5. `ecup_matching/experiments/v6/PLAN.md`
6. `ecup_matching/experiments/v6/RESULTS.md`
7. `ecup_matching/experiments/v6/SAFE_METRICS.json`
8. `docs/agent-memory/SECURITY.md`
9. `docs/agent-memory/ITERATION_PROTOCOL.md`
10. `ecup_matching/SOLUTION_RESEARCH.md`
11. `ecup_matching/BASELINE_CONTRACT.md`

Historical v1-v5 plans/results remain useful for evidence and failure lessons but do not redefine the current v6 runtime-selection rule.

## Validation invariants

- Never alter split SHA `aae58f...eb55b` to improve a metric.
- Never inspect/use sealed-gold labels during development or runtime tuning.
- Do not encode/mine sealed-gold items as adaptation, hard-negative or weak-label data.
- Every target-fitted model or stack must score held components using parameters fit without those held labels.
- Production refit on all development labels is allowed only after selection and must never be reported as validation.
- `0.6000` means honest strict development OOF, not a repeatedly tuned holdout score.
- Public/Private leaderboard scores are a separate evidence axis and must not overwrite local OOF.

## Current v6 architecture

Retained non-teacher signals:

1. weak category specialist;
2. sparse TF-IDF specialist;
3. explicit per-key attribute specialist;
4. supervised contrastive item score;
5. typed/canonicalized explicit specialist.

Target-free percentile-rank disagreement among these signals selects the highest-disagreement 95% of pairs inside each category for real pair-teacher inference. The remaining teacher signal is the mean of the five non-teacher percentile ranks. Final meta is category-shrunk simplex + fixed HGB, fused with frozen 50/50 percentile ranks under full outer cross-fitting.

## v6 runtime rules

Current FP32 implementation uses semantic-preserving speedups first:

- length-bucket contrastive item texts;
- length-bucket selected teacher pairs;
- VRAM-aware CUDA batches; 8 GiB default `256` contrastive / `96` teacher;
- CUDA OOM batch-halving fallback;
- non-blocking transfers;
- SDPA where supported with eager fallback;
- offline/local-files-only inference;
- phase telemetry for load, structured, contrastive, gate, teacher, meta and write.

Do not enable mixed precision, quantization, shorter max lengths or a materially different model merely because it is faster unless its resulting predictions are separately validated against the strict `>=0.6000` gate.

## Retained / rejected runtime evidence

KEEP:

- v5 quality reference: `0.6018115534135564`;
- v6 teacher gate 95%: `0.6006003614522999`.

REJECT under the current quality gate:

- structured only `0.5808404005946962`;
- no teacher `0.5931387077244183`;
- no contrastive `0.5928725263319000`;
- teacher gate 25% `0.5929214688140778`;
- teacher gate 55% `0.5966896566149946`;
- teacher gate 85% `0.5999300791828578`;
- teacher distillation `0.5931935841654697`;
- student + real teacher 85% `0.5998746122650258`.

Infrastructure, OOM, runner, packaging and API failures are not model-quality evidence.

## Production completion gate

A v6 submission is not completed until the same architecture has:

1. selected-contract tests GREEN;
2. deterministic production meta refit;
3. verified source/base artifact hashes;
4. exact organizer-image offline/read-only smoke;
5. full repository tests and `scripts/memory_policy.py` GREEN;
6. final ZIP SHA/provenance;
7. exact-byte benchmark on `ecup-rtx2060` inside the organizer image, including a full reference `matches.parquet` run;
8. private HF upload and GitHub Actions artifact of the exact retained ZIP;
9. documentation updated with final runtime/hash/artifact evidence.

Never weaken a test gate to publish a package.

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

Current checkpoint target after v6 completion is `--iteration v6`.
