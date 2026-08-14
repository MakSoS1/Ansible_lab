# E-CUP v5 Fixed Nonlinear HGB Meta-Stack Gate

Date: 2026-08-11

## Frozen entry state

- six-signal equal-rank OOF Macro AP: `0.5975445721449741`;
- fully outer-cross-fitted global simplex OOF Macro AP: `0.5992720660193247`;
- category-logistic standalone OOF: `0.5988060044248327` and therefore not the current best;
- immutable split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- sealed gold remains unopened, `0` rows scored.

## Frozen candidate

Train one shallow nonlinear meta-model on the same six target-free global percentile-rank inputs plus official category as a categorical feature.

For each immutable outer fold:

1. compute the six percentile-rank inputs target-free;
2. encode the 20 official categories with one deterministic global vocabulary (category names only, no labels);
3. exclude the outer fold completely from fitting;
4. fit `sklearn.ensemble.HistGradientBoostingClassifier` on the other four folds using category-balanced sample weights;
5. predict probability only for the excluded fold.

Hyperparameters are frozen before seeing this candidate's real OOF result:

- `learning_rate=0.05`;
- `max_iter=160`;
- `max_leaf_nodes=15`;
- `max_depth=3`;
- `min_samples_leaf=200`;
- `l2_regularization=5.0`;
- `early_stopping=False`;
- `random_state=20260811`;
- category column is marked categorical;
- no class-threshold tuning, no feature subset search, no parameter grid, no post-result calibration.

The model is trained independently in each outer fold. Since all feature transformations are target-free and hyperparameters are fixed, the held-out fold's labels cannot affect its prediction.

## Gate

KEEP only if strict-official development Macro AP exceeds `0.5992720660193247`, with sealed gold still unopened. The engineering milestone is reached at strict-official OOF Macro AP `>= 0.60`.

If it misses, do not tune these HGB hyperparameters on the same aggregate OOF result; move to the already frozen normalized-retrain evidence sources or another separately predeclared representation/model family.
