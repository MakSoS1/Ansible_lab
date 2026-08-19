# E-CUP v5 Normalized Retrain Fusion Gate

Date: 2026-08-11

## Frozen entry state

Current clean target-free development best is **`0.5975445721260434`** from six equal global percentile-rank signals:

1. leakage-safe weak specialist;
2. sparse TF-IDF specialist;
3. old explicit attribute specialist (`cb350b4e7ba6`);
4. raw held-out supervised-contrastive cosine;
5. raw held-out field-aware teacher2 score;
6. typed-explicit specialist (`ccc717a53fc1`).

All five folds improved versus the prior `0.5952697490140912` anchor. Gap to `0.60` is `0.0024554278739566`.

This candidate set is frozen **before** the new normalization-triggered explicit/category retrains finish.

## New sources

- `normalized_explicit`: the explicit specialist retrained after both typed quantity/value canonicalization and separator-insensitive model/SKU normalization (`SM-S921B == sms921b == SM_S921B`). Same HGB hyperparameters and same outer folds.
- `normalized_category`: the category-specialist structured model retrained on the same latest text/feature normalization. Same architecture/hyperparameters and same immutable v5 split.

## Exactly four predeclared target-free candidates

Let `current6` denote the six frozen signals above.

1. `current6_plus_normalized_explicit`: current6 + normalized_explicit as a seventh equal global percentile-rank vote.
2. `current6_replace_typed_with_normalized_explicit`: weak + sparse + old explicit + contrastive + teacher2 + normalized_explicit.
3. `current6_replace_old_explicit_with_normalized_explicit`: weak + sparse + normalized_explicit + contrastive + teacher2 + typed_explicit.
4. `current6_plus_normalized_category`: current6 + normalized_category as a seventh equal global percentile-rank vote.

No learned coefficients, duplicated weighted votes, category-specific weights, threshold tuning, source subsets beyond these four, or post-result extra candidates are permitted in this gate.

## Gate

A candidate is KEEP-eligible only if:

- strict-official development Macro AP > `0.5975445721260434`;
- minimum held-fold delta vs current6 >= `-0.001`;
- `target_fitted_blender=false`;
- sealed-gold metrics/scores remain uninspected.

The engineering milestone is reached when aggregate development OOF is `>= 0.60`.

If none reaches `0.60`, close representation-retrain fusion and move to a genuinely new supervised evidence source (true outer-isolated stack / stronger pairwise teacher), not manual rank-weight tuning.
