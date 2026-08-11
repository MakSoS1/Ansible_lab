# E-CUP Matching — Iteration v5 Plan

Date: 2026-08-11
Status: **in progress**

Canonical design: `docs/superpowers/specs/2026-08-11-ecup-v5-target-proxy-contrastive-residual-design.md`.

Implementation plan: `docs/superpowers/plans/2026-08-11-ecup-v5-target-proxy-contrastive-residual.md`.

## Why v5 exists

Leaderboard evidence invalidated the old assumption that the repeatedly reused 73,131-row validation predicts hidden performance. Observed hidden Macro AP supplied by the participant:

- v1: `0.23458522924335687`;
- v2: `0.2583231811423486`;
- v3 non-canonical: `0.2583231811423486`;
- v3 canonical: `0.24810151893254498`;
- v4 canonical: `0.2531285194869718`.

v2 is therefore the immutable production anchor until a new candidate passes the v5 validation gate.

## v5 goals

1. Replace the single adaptively reused holdout with balanced component-disjoint five-fold development validation plus a sealed gold holdout.
2. Keep gold labels unread during architecture/hyperparameter development.
3. Introduce genuinely new item-level representations rather than another pairwise Cross-Encoder/blend copy.
4. Treat v2 as the anchor and learn conservative residual corrections.
5. Expand weak-label usage with soft probabilities and streaming selection.
6. Reach the user-requested stretch target of honest local Macro AP `>= 0.600000` without fitting or selecting on sealed-gold labels.

## Current ladder

- `v5a-validation-audit`: completed; gold remains sealed.
- `v5b-sparse/residual`: implementation and OOF evaluation in progress.
- `v5c-pretrained-biencoder`: next; symmetric item encoder, no pairwise fine-tuning.
- `v5d-contrastive-human`: supervised item-space fine-tuning if pretrained item embeddings transfer.
- `v5e-contrastive-weak`: 2–4M soft weak-label curriculum.
- `v5f-hard-mined/teacher-distilled`: only if earlier rungs justify additional complexity.

## Gold rule

The 80,444-row gold partition created by v5a is not scored during development. It may be read only after candidate configuration/checkpoint/preprocessing hashes are frozen. If a gold evaluation is eventually performed, its result is evidence, not a new hyperparameter tuning surface.
