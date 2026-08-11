# E-CUP v5 Orthogonal Rank Fusion Ablation

Date: 2026-08-11

## Entry evidence

Current clean target-free development best is `global_rank_mean_4_no_category = 0.5870570848443828`, produced from equal global percentile ranks of:

- weak specialist;
- sparse TF-IDF specialist;
- explicit attribute specialist;
- held-out supervised-contrastive raw cosine.

All five held folds improved against the prior target-free anchor. Gap to `0.60` is `0.0129429151556172`.

Field-aware weak teacher run `31486298300` has now completed. Its direct raw score has low standalone AP (`0.42454568571958984`), but standalone AP is not a sufficient reason to reject an orthogonal rank signal: the supervised-contrastive raw cosine is also weak standalone yet materially improves equal-rank fusion. Teacher2 fitted-stack output is not eligible for this clean experiment because it uses the same non-nested second-level pattern discussed in the transfer-safe design.

## Frozen new evidence sources

1. `teacher2_raw`: `teacher2_score` from five true held-out teacher2 fold files under source `411a5349fe73`.
2. `weighted`: direct OOF weighted-category-specialist score from source `9df24f7ee133`.
3. `pretrained_raw`: raw `embedding_cosine` from pretrained multilingual item-space source `d812ad8d6a00`.

Every source must align exactly to the frozen development `row_index` and fold manifest before scoring.

## Predeclared candidate set

Let `current4 = weak + sparse + explicit + contrastive_raw`, with equal global percentile-rank voting.

Evaluate exactly these five additional target-free candidates:

1. `current4_plus_teacher` = current4 + teacher2_raw;
2. `current4_plus_weighted` = current4 + weighted;
3. `current4_plus_pretrained` = current4 + pretrained_raw;
4. `current4_plus_teacher_weighted` = current4 + teacher2_raw + weighted;
5. `current4_plus_all_three` = current4 + teacher2_raw + weighted + pretrained_raw.

No target labels, learned coefficients, category-specific weights, grid search, or post-result source dropping may be used inside this ablation.

## Gate

Comparison anchor: `0.5870570848443828`.

A candidate is KEEP-eligible only if:

- strict-official aggregate Macro AP > anchor;
- minimum held-fold delta vs anchor >= `-0.001`;
- `gold_metric_opened=false`, `gold_rows_scored=0`;
- `target_fitted_blender=false`.

If all candidates fail, close this equal-rank orthogonal-source branch. Do not tune continuous weights on these development rows. Move to genuinely new evidence: typed numeric/attribute normalization and/or a true outer-isolated supervised stack.
