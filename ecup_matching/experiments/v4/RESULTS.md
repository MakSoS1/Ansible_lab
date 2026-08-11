# E-CUP Matching — Iteration v4 Results

Date: 2026-08-11
Status: **in progress**

## Baseline

Retained v3 Macro AP: `0.5254642645846543` on the fixed 73,131-row item-disjoint validation with zero train/validation item overlap.

Immutable v3 submission SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`.

## Planned measured candidates

- v4a: `ai-forever/ruBert-base` + complete human curriculum.
- v4b: v4a continuation + confidence-filtered LLM weak curriculum.
- v4c: best v4a/v4b continuation + model-mined hard negatives with 50% ordinary replay.

No v4 quality metric is recorded here until a real run has produced and passed the complete fixed-validation checks.

## Execution evidence

### RTX production attempt 1 — rejected infrastructure run

Private dispatcher run: `31470932265`, job `93713870740`.

Public source SHA: `b279b4cb7749e958a8f773b45d9972194e6c1cc8`.

Observed facts:

- trusted v4 image was available and `ai-forever/ruBert-base` loaded successfully;
- the classifier head initialized as expected;
- the isolated training process terminated with exit code `137` before a training/validation metric was produced;
- GitHub runner logs also recorded a runner shutdown signal;
- no `metrics.json` was returned, no v4 stage was retained, and no submission ZIP was promoted.

Interpretation: this is **not a negative ML result and not a measured CUDA OOM**. The process died in the memory-heavy preparation path before model-selection evidence existed.

Root causes removed after this run:

1. the structured weak-label presample previously materialized the >11M-row LLM parquet in pandas; the current production path scans parquet batches with `pyarrow.iter_batches` and keeps only the bounded deterministic presample;
2. the 178M-parameter BERT was previously materialized before the structured/data preparation; current v4 prepares structured/curriculum inputs first, releases transient allocations, then loads BERT;
3. weak serialized pairs now use a temporary canonical key for dedup/conflict filtering so reversing a key can never detach `id1/id2` from `text_a/text_b`;
4. v4 inference batch size is VRAM-aware: 8 GiB RTX uses a conservative batch while the organizer H100 can use the larger batch.

The bounded-memory public state passed the regular repository CI and memory-policy gate before the replacement production run was queued.

### RTX production attempt 2 — queued/recovery

Private dispatcher run: `31472201045`.

Public source SHA: `6420b64f60c04e91d95d6092acbae876df2f1d34`.

This run uses the bounded-memory preparation path. At the latest recorded checkpoint the self-hosted runner had not picked up the job, so no metric is attributed to it yet.

### v4a MPS diagnostic — in progress

GitHub-hosted Apple Silicon diagnostic run: `31472651366`.

Purpose: independently test whether the exact pinned stronger encoder has enough quality upside on the complete leakage-safe human pool while the home runner is unavailable. It uses the unchanged 73,131-row item-disjoint validation and the frozen v2b validation anchor. It is diagnostic evidence only and cannot substitute for the canonical RTX v4b/v4c ladder unless its exact training budget and final packaging gates are separately accepted.

## Current decision

v3 remains current best until a v4 candidate strictly exceeds `0.5254642645846543` and passes the packaging/runtime gates. No failed infrastructure run is treated as a v4 score.
