# E-CUP Matching — Canonical Project State

Updated: 2026-08-12
Current iteration: **v7**

## Objective

Maximize E-CUP 2026 product-matching Macro AP while preserving honest unseen-product validation and producing an offline organizer-compatible solution that fits the competition runtime constraint.

Current v7 ordering is:

1. preserve the immutable five-fold component-disjoint validation and unopened sealed gold;
2. create a materially stronger pairwise neural signal than the retained v5/v6 teacher path;
3. beat the v5 strict quality reference `0.6018115534135564`; `0.70` is the stretch target, never a number to claim without evidence;
4. retain only architectures that also pass the production runtime gate.

## Current state — read this first

- **Best strict local quality reference:** v5, `0.6018115534135564` OOF Macro AP.
- **Best current runtime-constrained reference:** v6 gate95, `0.6006003614522999` strict OOF.
- **Current experiment:** v7 identity-first full-context pair teacher on branch `ecup-v7-neural`.
- v7 base encoder: `ai-forever/ruBert-base`, revision `43be4261797042e172adf7476c558734f3cbb2a0`.
- v7 changes the pairwise signal hypothesis, not the frozen validation: 256-token pair context, identity/typed attributes before numeric tail, and more leakage-safe curriculum exposure.
- The retained v5 `teacher2` already used ruBERT-base. Its limiting settings were `max_length=128`, `max_steps=800`, at most 100k weak rows per fold, and v5 serialization that placed generic `[NUMERIC]` before `[ATTR]`.
- v7 RED contract was observed in Actions run `31546090474`; after implementing isolated v7 serializer/safety modules, the v7 targeted code tests passed in run `31546214410`. That run then failed only because canonical memory documents still described v6; those documents are now being updated before training.
- **No v7 OOF score exists yet.** `0.70` is a target only.
- Sealed gold: **unopened**, `0` rows scored.
- Public/private leaderboard score for the retained local candidates: **unknown**. Previous platform submissions failed the time limit before scoring.

## Immutable validation protocol

- human labels: `365,654` rows;
- connected item components: `345,654`;
- development rows: `285,210`;
- sealed-gold rows: `80,444`;
- five immutable development folds;
- cross-split item/component overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- strict metric: `average_precision_score` per official category, unweighted mean over exactly 20 categories.

Every target-fitted layer must be genuinely outer-cross-fitted. Full-development production refits are not validation. Sealed-gold labels are unavailable for architecture choice, runtime tuning, mining or calibration.

For a common weak pretraining checkpoint reused by all outer folds, the weak corpus must exclude the complete human-item universe from the immutable split. Otherwise weak training must remain fold-specific and exclude the held/gold item universe for that fold.

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

The v5 ladder shows diminishing returns from further lexical/meta additions: category `0.547678`, weak `0.551424`, sparse `0.565131`, explicit `0.568307`, contrastive `0.566222`, four-signal rank `0.587057`, + teacher `0.595270`, six-signal `0.597545`, outer meta stages up to final `0.601812`. That is why v7 targets the neural pair model rather than another weight search.

## v6 runtime reference

The selected v6 candidate uses the real pair teacher on target-free highest-disagreement 95% of pairs per category, with a five-signal rank surrogate elsewhere. Strict OOF is `0.6006003614522999`; lower-cost variants did not clear `0.6000`.

Retained prediction-preserving runtime work:

- structured feature chunking pinned at `10,000` and distributed over a `fork` worker pool without re-chunking;
- shared `difflib` ratios between legacy and typed passes;
- shared `ItemNorm` construction where possible;
- single-pass `select_items_by_ids`;
- deferred CUDA initialization;
- stable length bucketing, VRAM-aware batches, OOM fallback, non-blocking transfers, SDPA/eager fallback;
- offline/local-files-only inference and phase telemetry.

Measured local structured profile: `2210.1 -> 487.0 us/pair`, `4.54x`, bitwise identical under the measured contract. Projected structured phase became approximately `~22s` public / `~44s` private instead of `~254s` / `~608s`. These are projections, not the final exact-byte end-to-end timing claim.

The authoritative runtime benchmark still requires the exact final production path on a full reference `matches.parquet`. A 64-row CPU smoke is compatibility evidence only.

## v7 Candidate A

### Diagnosis

The retained pair teacher is the only signal that jointly reads both item texts, but its contribution was much smaller than expected. The concrete architecture issues now under test are:

- pair context only 128 tokens;
- generic numeric material allocated before canonical explicit attributes;
- only 800 optimizer steps;
- weak exposure capped at 100k rows per fold;
- only the tail of the encoder trainable in teacher2.

### Implemented contract layer

New v7 modules are isolated so v5/v6 behavior stays reproducible:

- `ecup_matching/ml/v7_item_text.py` — deterministic identity-first serializer with hard character bounds;
- `ecup_matching/ml/v7_teacher_contract.py` — 256-token minimum, curriculum-step validation and explicit forbidden-item weak filtering;
- `ecup_matching/tests/test_v7_neural_contract.py` — canonical `128 GB == 0.128 TB`, model-code visibility, deterministic hard bounds, forbidden endpoint filtering and legacy `128/800` rejection.

Candidate A training design:

1. identity packet: name, brand, normalized model/SKU, canonical typed values;
2. residual numeric and low-priority attributes only after the identity packet;
3. ruBERT-base at `max_length=256`;
4. leakage-safe confidence-weighted weak exposure plus authoritative human fine-tuning;
5. five outer-fold held predictions only for strict OOF;
6. progress/timing telemetry with phase, done/total, throughput, ETA, RAM and CUDA information where available.

Candidate B exists only if A is insufficient: an aligned shared-key pair view. It must independently win strict OOF/runtime gates.

## GPU dispatch

Home GPU remains isolated in private `MakSoS1/gpu-dispatch` on `ecup-rtx2060` (RTX 2060 SUPER, 8 GiB). The current trusted dispatcher still pins source authorization to the historical `ecup-matching-2026` branch and its fixed profiles are v4-era. Before v7 training, the private dispatcher must be extended narrowly for the dedicated v7 branch/profile rather than granting arbitrary commands or moving public runner ownership.

The RTX 2060 is the training/feasibility device; the competition inference host is materially stronger (H100 80 GiB). Runtime decisions therefore need actual measured neural throughput plus the organizer limits, not a requirement that the 2060 itself match H100 wall time.

## Binding failure lessons

- Infrastructure, OOM, packaging or API failures are not model scores.
- Do not weaken full tests to publish an artifact.
- Production refit scores are not validation.
- Do not use sealed gold to recover a runtime-induced quality loss.
- Direct attribute score shifts failed while explicit per-key estimator features were useful; do not conflate them.
- Pretrained-only embeddings were weak; supervised contrastive was the useful neural signal.
- The final teacher already uses ruBERT-base; a model-name swap alone is not a new hypothesis.
- Any mixed-precision/quantized inference path that can alter ordering requires its own honest quality verification.
- A fixed-overhead smoke is not runtime evidence.
- The submission file list must be derived from the import graph, never hand-maintained.
- Structured chunk size is not a free parameter because float32 GEMM batching can perturb scores.
- Never report a target such as `0.70` as achieved before the five-fold aggregate exists.

## Current files to read

1. `ecup_matching/experiments/CURRENT.json`
2. `ecup_matching/experiments/v7/PLAN.md`
3. `ecup_matching/experiments/v7/RESULTS.md`
4. `ecup_matching/experiments/v7/SAFE_METRICS.json`
5. `docs/agent-memory/EXPERIMENT_INDEX.md`
6. `docs/agent-memory/DECISIONS.md`
7. `docs/agent-memory/SECURITY.md`
8. `docs/agent-memory/ITERATION_PROTOCOL.md`
9. retained v6 plan/results/runtime evidence

## Next action

Make the v7 repository/memory gate GREEN, extend the private trusted GPU dispatcher with a fixed v7 profile, measure real ruBERT-base 256-token feasibility on the RTX 2060, then run Candidate A under the immutable five-fold OOF contract. Record every KEEP/REJECT/FAIL in v7 results/safe metrics and Memora. Do not open sealed gold.