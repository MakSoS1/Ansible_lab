# E-CUP v5 Transfer-Safe Stack Design

Date: 2026-08-11

## Goal

Continue toward honest development Macro AP >= 0.60 without opening sealed gold, while correcting evaluation assumptions that can make a stacked OOF score look cleaner than it really is.

## Frozen constraints

- Keep split SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b` unchanged.
- Development rows: 285,210; sealed gold: 80,444; five component-disjoint folds; item overlap 0.
- Do not score sealed gold during development.
- v2 remains production/hidden fallback until a frozen v5 candidate passes transfer/runtime gates.
- Current clean direct development anchor is explicit per-key attribute specialists: `0.5683065131240066`.

## Evaluation correction

The existing `crossfit_semantic_stack()` correctly excludes each row's target from the second-level estimator that predicts that row, but this is not sufficient for a fully nested evaluation when its input features are themselves supervised OOF predictions generated with the same folds.

For an outer held fold `j`, a meta-training row in fold `k != j` has a base OOF prediction from a base model trained on all folds except `k`; that base model can include labels from `j`. Therefore the held fold can influence meta-training feature values indirectly. This is second-order fold leakage.

Consequences:

- direct category/weak/sparse/explicit held-fold scores remain valid held-out branch measurements;
- raw contrastive semantic features for a held row remain held-out when produced by the encoder trained without that fold;
- fitted second-level scores based on the precomputed same-fold OOF matrix are diagnostic, not fully nested evidence;
- `0.5595125314` combo and `0.5662217063` contrastive stack must not outrank the direct explicit `0.5683065131` as the clean development anchor until re-evaluated with outer-isolated features.

## Phase A: fixed label-free strong-signal blend

Use only already-held-out branch outputs and no target-fitted blender:

- category specialist score;
- weak specialist score;
- sparse specialist score;
- explicit attribute specialist score;
- raw held-out supervised contrastive cosine.

Evaluate a small, predeclared set of label-free fusion rules:

1. arithmetic mean of branch percentile ranks;
2. arithmetic mean of clipped probabilities for the four calibrated classifier scores;
3. rank mean of explicit+sparse+weak with category as a stabilizing fourth vote;
4. rank mean of explicit+sparse+weak+category+contrastive cosine.

The workflow reports every rule, per-fold AP, per-category AP, and deltas vs explicit. It does not fit weights from targets. A rule is KEEP only if aggregate AP improves explicit and no fold has material regression (>0.001 absolute).

## Phase B: metric contract hardening

Competition evaluation must fail loudly when used in official/full-development mode if:

- the category set differs from the exact 20 official categories;
- any category has only one target class;
- scores are non-finite.

Toy/unit callers may continue using generic category sets through the existing non-strict API. v5 full-development reports use strict official mode.

## Phase C: true outer-isolated supervised stack (only if Phase A < 0.60)

For each outer fold, train/generate every supervised feature using only outer-train rows, generate meta-training features without touching outer-valid labels, fit the meta-model on outer-train, and score outer-valid. Existing same-fold OOF matrices are insufficient for this proof.

Preferred implementation order for cost:

1. category + explicit + sparse;
2. add weak;
3. add supervised contrastive semantic features;
4. only then add a fully trained pairwise Transformer teacher.

## Transfer proxy

Do not change the frozen v5 scoreboard. Add a separate transfer track later: signature-disjoint stress validation plus rare diagnostic Public submissions. Development AP, transfer-stress AP and Public score are separate fields.

## Documentation and memory

Every KEEP/REJECT/FAIL updates `RESULTS.md`, `SAFE_METRICS.json`, `CURRENT.json` when needed, `EXPERIMENT_INDEX.md`, and durable decisions. Gold wording should mean `gold metrics/scores never inspected after split freeze`; target labels were used once to stratify the frozen split and this is not model leakage.
