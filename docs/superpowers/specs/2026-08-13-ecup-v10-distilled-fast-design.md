# E-CUP v10 distilled-fast design

Date: 2026-08-13

## Problem

v8, v9, and compact-v9 local runtime gates did not transfer to the competition platform: the platform still returned `Container did not finish in time`. Therefore v10 must not incrementally optimize the v9 multi-stage runtime. It must change the inference architecture.

## Decision

Use expensive v9 components only as offline teacher evidence. The submitted archive contains one small pair cross-encoder student plus an optional target-free O(N log N) graph/rank postprocess. No v9 teacher, contrastive encoder, structured model, sparse model, or HGB stack may run inside the submission.

Base student candidate: `cointegrated/rubert-tiny2`, pinned by immutable revision during training/build. Preserve the v7 identity-first typed text serialization initially, then reduce `max_length` / `max_chars` only when held-out evidence supports the change.

## Validation

Use the immutable component-disjoint development split only:

- 285,210 development rows;
- 80,444 sealed gold rows, never scored;
- five outer folds;
- split SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- unweighted Macro AP over the 20 official categories.

Do not accept a fold-0-only result as v10 quality evidence. Distillation used for OOF must be fold-safe: a held fold cannot be trained against teacher values produced by a model fitted on that held fold. If fold-safe teacher values are unavailable, the corresponding validation run is hard-label-only rather than pretending to be distilled.

## Training objective

Student BCE target is a convex combination of the human hard label and an optional fold-safe teacher probability/rank target. Category-local ranking loss continues to use the human hard label so teacher softness cannot erase the official binary ordering objective. Sampling remains category-balanced. Training-only weak/hard-negative data may be used if it has zero human-item overlap.

## Runtime contract

The submitted runtime is student-only. Internal acceptance is deliberately much stricter than the organizer nominal budget because local RTX timing failed to predict platform timing for v9.

- 275k full end-to-end target: <= 220 s;
- hard reject: > 250 s;
- timer starts before safe ZIP extraction and ends after output validation;
- package should be far below v9 size and contain no unused heavyweight checkpoints.

The 220/250 s thresholds are internal engineering gates, not organizer-published limits.

## Selection

Compare at least:

1. tiny student hard-label baseline;
2. fold-safe distilled tiny student when teacher artifacts permit it;
3. training-only weak/hard-negative augmentation;
4. max-length / max-chars runtime-quality Pareto choices;
5. optional frozen target-free graph rescore.

Select before production refit. Production refit is not validation.

## Release contract

A v10 keeper may be published only after:

- five-fold OOF evidence is complete;
- target-stress is recorded separately from strict OOF;
- exact package passes schema/order/finite-score checks;
- exact package passes the <=250 s private-size internal runtime gate;
- full repository tests and memory policy pass;
- exact SHA-256 is frozen and uploaded to private HF;
- canonical state and Memora checkpoint are updated without rewriting v7-v9 history.
