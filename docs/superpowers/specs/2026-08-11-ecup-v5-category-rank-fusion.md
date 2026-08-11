# E-CUP v5 Category-Rank Fusion Ablation

Date: 2026-08-11

## Evidence entering this ablation

The first predeclared target-free fusion run (`31492628448`, source `c78b245e092a`) produced:

- explicit direct anchor: `0.5683065131240066`;
- global `rank_mean_5`: `0.584345401516843`;
- delta vs explicit: `+0.016038888392836403`;
- all five held folds improved by at least `+0.01514`;
- no target-fitted blender was used.

This leaves `0.015654598483157` to the `0.60` development milestone.

## Metric-alignment observation

The competition metric computes AP independently inside each of the 20 categories and averages those 20 AP values. Ordering scores across different categories is irrelevant to the metric. Therefore a global percentile transform spends scale resolution on cross-category comparisons that never enter AP.

A category-local percentile transform is target-free, preserves every source's within-category ordering, is directly reproducible on an offline test candidate batch, and aligns fusion calibration with the metric's comparison domain.

## Frozen candidate set

This ablation is declared before observing its metrics. It evaluates exactly five new target-free fusion rules:

1. `global_rank_mean_4_no_category` = weak + sparse + explicit + contrastive cosine;
2. `global_rank_mean_3_strong` = sparse + explicit + contrastive cosine;
3. `category_rank_mean_5` = category + weak + sparse + explicit + contrastive cosine, with each source ranked within category;
4. `category_rank_mean_4_no_category` = weak + sparse + explicit + contrastive cosine, ranked within category;
5. `category_rank_mean_3_strong` = sparse + explicit + contrastive cosine, ranked within category.

No learned weights, target labels, per-category coefficient tuning, or grid search are allowed in candidate construction.

## Selection gate

The comparison anchor is the previous global `rank_mean_5 = 0.584345401516843`.

A candidate can become the new development best only if:

- aggregate strict-official Macro AP is greater than `0.584345401516843`;
- every held fold remains non-regressive within `-0.001` vs the global rank_mean_5 anchor;
- the runner confirms `gold_metric_opened=false`, `gold_rows_scored=0`, and `target_fitted_blender=false`.

If none reaches 0.60, do not tune continuous weights on the same rows. Continue to new orthogonal evidence (typed normalization / outer-isolated supervised stack / completed teacher).
