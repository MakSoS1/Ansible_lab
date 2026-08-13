# v10 Plan — distilled fast single-student submission

Status: **IN PROGRESS**

Date: 2026-08-13

## Why v10 exists

The v9 multi-stage architecture and its compact FP16-storage variant both failed on the competition platform with `Container did not finish in time`, even though the local RTX 2060 SUPER gate used the organizer image and measured extraction + inference + validation. Therefore local v9 timing is retained as engineering evidence but is no longer treated as a reliable proxy for platform completion.

v10 is not another v9 compression pass. It changes the inference architecture.

## Runtime architecture

Submission runtime must contain exactly one small pair cross-encoder student. Current baseline candidate:

- base model: `cointegrated/rubert-tiny2`;
- pinned revision: `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`;
- pair serialization: v7 identity-first typed text;
- baseline `max_length=128`, `max_chars=650`;
- no v9 teacher inference;
- no contrastive encoder inference;
- no structured/TF-IDF/HGB inference;
- optional graph rescore only if it improves every outer fold and preserves the runtime gate.

The expensive historical models may be used only as training/research evidence when leakage-safe. They do not belong in the v10 submission archive.

## Immutable validation

- development rows: `285,210`;
- sealed gold rows: `80,444`;
- five component-disjoint outer folds;
- cross-split item overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- metric: unweighted Macro AP over exactly 20 official categories;
- sealed gold stays unopened.

A fold-0 diagnostic is not acceptable selection evidence. Production refit is never validation.

## Distillation rule

Distillation is retained as a v10 capability, not an automatic claim. Strict OOF may use a teacher only when that teacher did not fit labels from the current outer validation fold. Existing ordinary OOF teacher predictions are not silently reused as nested fold-safe distillation targets.

If no leakage-safe teacher artifact is available for a candidate, that OOF candidate remains hard-label-only.

## Runtime gate

Because v9 local timing failed to transfer, v10 uses a much larger safety margin:

- target private-size E2E: `<=220 s`;
- hard reject private-size E2E: `>250 s`;
- timer starts before safe ZIP extraction and stops after output validation;
- exact organizer image: `odsai/ecup26-matching-baseline:1.0`;
- final package must contain exactly one model weight file.

These are internal acceptance limits, not organizer-published limits.

## Early runtime evidence

RTX 2060 SUPER throughput run `31679631580` tested the pinned tiny model. At `max_length=128`, batch `512`, pure pair scoring was approximately `2,216 pairs/s`, projecting about `124 s` for 275k scoring. This is a throughput probe, not the final package E2E proof; the final ZIP still must pass extraction + inference + validation under the hard `250 s` gate.

Later sweep points were warmer and faster, so they are not used as the conservative runtime claim.

## Research order

1. complete honest five-fold tiny baseline OOF;
2. evaluate frozen target-free graph fold-locally;
3. only if quality needs improvement, scout student-only training changes (sequence budget, epochs/learning rate, weak/hard-negative curriculum, leakage-safe distillation);
4. run complete five-fold OOF for the selected quality candidate;
5. production refit on all development rows;
6. build minimal exact ZIP;
7. exact 115k/275k E2E runtime gate with private hard reject `250 s`;
8. full repository verification;
9. publish exact SHA to private HF;
10. update CURRENT/SAFE_METRICS/RESULTS and hardened Memora checkpoint.

## Historical evidence separation

- v7 owner-reported leaderboard `~0.36` remains an external anchor only;
- v9 strict/graph/target-stress metrics remain historical local evidence;
- neither v9 local metric nor v9 local runtime is rewritten after the platform timeout;
- v10 leaderboard remains unknown until the platform actually scores the exact v10 archive.
