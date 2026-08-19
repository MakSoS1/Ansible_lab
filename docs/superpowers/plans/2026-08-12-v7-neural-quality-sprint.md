# E-CUP v7 Neural Quality Sprint Implementation Plan

**Goal:** Raise strict component-disjoint OOF Macro AP materially above the retained 0.6018115534 quality reference, with 0.70 as the stretch target, while preserving sealed-gold isolation and proving runtime feasibility under the organizer limits.

**Starting point:** Branch `ecup-v7-neural` from `ecup-v6-fast-runtime` at `0580eeed2fb04f363951a5a325442430e4639e0c`. v6 runtime optimizations are prediction-preserving. The best retained strict OOF quality is v5 `0.6018115534135564`; current fast v6 gate95 is `0.6006003614522999`.

**Validation invariant:** immutable split SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`, 285,210 development rows, five component-disjoint outer folds, zero cross-split item overlap, exactly 20 category AP values averaged without weights. Sealed gold remains unopened.

## Evidence-driven diagnosis

1. Final v5 already uses `ai-forever/ruBert-base`; changing only the model name cannot explain a large gain.
2. The retained teacher2 was constrained to `max_length=128`, `max_steps=800`, and at most 100k weak rows per outer fold.
3. `serialize_item_v5` orders `[NUMERIC]` before `[ATTR]`, so a long generic numeric section can evict canonical attributes from the cross-encoder context.
4. Existing structured/lexical/meta refinements have diminishing returns; a large gain requires a stronger, less correlated pairwise signal.

## Architecture

Create a v7 pair teacher around three changes that are independently testable:

- **Identity-first serialization:** `[NAME]`, `[BRAND]`, `[MODEL]`, then high-value canonical attributes/typed quantities, then residual numeric evidence and low-priority attributes. Preserve deterministic hard character bounds.
- **Longer pair context:** 256 tokens for ruBERT-base, matching the organizer baseline contract and fitting the H100 inference budget subject to measurement.
- **Leakage-safe curriculum:** authoritative human fold training plus confidence-weighted weak rows that cannot contain any held-fold/gold item. Increase useful weak exposure and train by epochs/steps sufficient to cover the curriculum, not an arbitrary 800-step cap.

Candidate B, only if A is insufficient: aligned pair serialization for shared discriminative keys (`key: A || B`) as an additional cross-encoder view. It must earn its place by strict OOF gain and measured runtime.

## TDD / execution sequence

### 1. Establish RED tests
- Add tests proving identity-critical canonical values appear before residual numeric/attribute tails under a short character budget.
- Add tests proving model/SKU separator normalization and typed-equivalent quantities serialize identically enough for the neural input (`128 GB` vs `0.128 TB`).
- Add leakage tests for weak-row filtering against the complete forbidden human validation item universe.
- Add training-plan tests that reject 128-token / 800-step legacy defaults for v7.

### 2. Implement identity-first serializer
- New module only; do not mutate v5/v6 serializer semantics.
- Deterministic attribute ranking: explicit learned priority when available, then identity-key heuristics, then stable lexical order.
- Canonicalize values through the existing `canonical_attribute_value` and quantities through existing typed extraction.
- Preserve name/brand/model at the front and hard-bound output.

### 3. Implement v7 outer-fold teacher
- Start from teacher2's correct outer-fold and weak-loss machinery, not the obsolete first teacher.
- Use ruBERT-base revision `43be4261797042e172adf7476c558734f3cbb2a0`.
- `max_length=256`; CUDA fp16 is allowed for training efficiency, but retained predictions must be produced by the declared inference path and revalidated.
- Build weak curriculum from rows whose two item IDs are outside the held-fold and sealed-gold forbidden set; keep source/confidence weights.
- Save held-fold OOF only; never score sealed gold.
- Emit progress telemetry: phase, done/total, elapsed, rolling throughput, ETA, RAM, CUDA VRAM/utilization where available, and timing JSON.

### 4. Strict aggregate and ablation
- Concatenate five held-fold OOF vectors with exact row identity checks.
- Report standalone teacher Macro AP and all 20 category AP values.
- Evaluate target-free/frozen blending against the existing retained six signals using outer cross-fitting only.
- KEEP only on strict comparable gain; otherwise REJECT and continue to candidate B.

### 5. Runtime gate
- Use private `MakSoS1/gpu-dispatch` on `ecup-rtx2060` for CUDA/training feasibility and exact-byte profiling where supported.
- Production inference benchmark must use the same serialized input, tokenizer settings, model weights and prediction path that will be packaged.
- Final runtime evidence must cover the full reference `matches.parquet`; smoke runs are not accepted as timing evidence.
- Organizer budgets: 360 s public, 780 s private. Keep a safety margin rather than merely fitting the limit.

### 6. Documentation / Memora
After every KEEP/REJECT/FAIL update:
- `ecup_matching/experiments/v7/RESULTS.md`
- `ecup_matching/experiments/v7/SAFE_METRICS.json`
- `ecup_matching/experiments/CURRENT.json`
- `docs/agent-memory/EXPERIMENT_INDEX.md`
- `docs/agent-memory/PROJECT_STATE.md`
- `docs/agent-memory/DECISIONS.md`

Then run `scripts/memory_policy.py`, ingest/checkpoint Memora, and keep raw data/checkpoints only in private storage.

## Completion gate

A v7 candidate is publishable only when all are true:
1. strict immutable-split OOF is reproduced and all five held folds are present;
2. sealed gold has not been opened;
3. quality is above the retained reference; 0.70 remains a stretch target, not a number to manufacture;
4. full tests and memory policy pass;
5. production refit is deterministic and packaged import closure is complete;
6. exact packaged bytes pass offline organizer-image smoke;
7. exact production path passes measured runtime gate with telemetry;
8. artifacts and hashes are persisted privately and documentation records the evidence.