# E-CUP v20 implementation record

Updated: 2026-08-19

## Frozen executable source

The queued private executor intentionally clones the exact public source SHA:

`2fdcbd62c291c9e354f052a2f0c980e2b25d71d6`

This documentation commit is post-freeze and is not part of that training SHA.

Private executor branch: `MakSoS1/gpu-dispatch:ecup-v20-executor`.
Queue-trigger commit: `59c64e382513099a8106f44c89287c813d8583b9`.
Concurrency: `ecup-isolated-gpu`, `queue: max`.

## Implemented D0-D10 path

- D0: machine-readable v1-v19 ledger and frozen policy/hash/leakage contracts.
- D1: streaming semantic census over full items + historical weak corpus with SQLite-backed item/weak-membership index; no full `items.parquet` pandas materialization.
- D2: never-labelled proxy from items that occur in neither human nor historical weak supervision; exact v7/v12/v13B/v14 archives must reproduce Public ordering `v14 > v12 > v13B > v7` or v20 fails closed.
- D3: component/item-disjoint human teacher audit for folds 0 and 1, two independent teacher lines, Wilson-LCB admission, no LLM confidence admission.
- D4: target-free real-item blocking, bounded item/reason degrees, two-teacher consensus, two-policy label intersection, active-review rejection path.
- D5: data-only RuBERT ablation on equal historical exposure.
- D6: rationale multi-task RuBERT ablation with training-only conflict/reason heads.
- D7: source-aware mixed replay with frozen Phase-B `1:2` human:other and Phase-C `4:1` at LR multiplier `0.35`.
- D8: scale only the selected causal mode from `1.2M -> 600k x 0.35` to `3M -> 1.5M x 1.0`.
- D9: matching historical controls + scaled candidate on folds 0 and 1; both folds must pass and mean human delta must be nonnegative.
- D10: full 285,210-row production refit, one `model_v7_teacher/*.safetensors`, proven v7-compatible packager, exact organizer-shaped Check, manifest/SHA-256, persistent final ZIP plus best-effort split Actions artifact.

## Teacher contract

Pinned teacher lines in the private job:

- `ibm-granite/granite-3.3-2b-instruct`;
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`.

At execution time each requested `main` revision is resolved to an immutable Hub commit SHA and recorded. A new LLM label is never admitted from one teacher, disagreement, `UNCERTAIN`, or deterministic-checker conflict. Generated labels used for training/production are the intersection of two independently calibrated fold policies.

## Current verification state

Implementation and queue wiring are complete. Do not claim GREEN tests, completed teacher audit, promoted v20 metrics, or a final ZIP until fresh Actions evidence exists. The first private `verify` job compiles every `v20_*.py` / `run_v20_*.py` file and runs every `tests/test_v20_*.py` inside the trusted image before any data/GPU stage. The public M1 smoke is supplemental; absence of a visible public run status is not treated as success.

## v18/v19 dependency rule

v20 does not assume v18 or v19 succeeded. It uses the same weak-quality path for its own matching control/candidates and selects against an exact historical v14 proxy anchor. The v19 refresh is applied only if persisted v19 two-fold confirmation explicitly says `promote=true`; otherwise v20 runs without it.
