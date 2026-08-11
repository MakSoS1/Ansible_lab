# E-CUP v5 Category Logistic Fusion Gate

Date: 2026-08-11

## Frozen entry state

- six-signal equal global percentile-rank OOF Macro AP: `0.5975445721449741`;
- fully outer-cross-fitted global simplex meta OOF Macro AP: `0.5992720660193247`;
- sealed gold remains unopened and has `0` scored rows;
- immutable split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.

The global simplex result improved every outer fold, but remained `0.0007279339806753` below the `0.60` engineering milestone.

## New target-fitted source

`category_logistic` is a fully nested logistic stack over the same six target-free percentile-rank inputs with global signal terms plus category-by-signal interaction terms.

For each outer fold:

1. that fold is completely excluded from target fitting;
2. `C` is selected by inner OOF over only the other four immutable folds;
3. category-balanced logistic loss is used so large categories cannot dominate the fit;
4. the selected model is refit on the four outer-train folds and scores only the excluded outer fold.

Therefore the aggregate `category_logistic` OOF vector is label-isolated at the outer-fold level, including hyperparameter selection.

## Frozen post-result fusion gate

Before inspecting the real category-logistic OOF metric, exactly one target-free fusion candidate is authorized if the standalone category-logistic score remains below `0.60`:

`global_meta_plus_category_logistic_equal_rank = 0.5 * percentile_rank(global_meta_oof) + 0.5 * percentile_rank(category_logistic_oof)`

No alternative fusion weights, category-specific fusion coefficients, thresholds, source subsets, or additional post-result rank mixtures are allowed in this gate.

The fusion is KEEP-eligible only if:

- strict-official development Macro AP is greater than `0.5992720660193247`;
- the aggregate strict-official development Macro AP is at least `0.60` for the engineering milestone;
- both component OOF vectors remain fully outer-cross-fitted;
- sealed-gold metrics/scores remain unopened.

If this frozen equal-rank fusion does not reach `0.60`, close this gate and proceed to a genuinely new representation/evidence source or a separately predeclared category-shrunk AP optimizer; do not tune fusion weights on the observed OOF labels.
