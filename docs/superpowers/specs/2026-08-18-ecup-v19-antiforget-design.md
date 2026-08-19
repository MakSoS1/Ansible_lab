# E-CUP v19 Anti-Forgetting Weak Refresh

## Motivation fixed before v19 results

The preregistered v17 x1 control run `32169719512` exposed a large training-order failure:

- weak holdout after weak phase: `0.6973930798915398` Macro AP;
- weak holdout after human phase: `0.6565798751243743` Macro AP;
- forgetting delta: `-0.0408132047671655`;
- human fold0 after human phase: `0.7017637363553895`.

The held weak split is item-disjoint from weak-train and has zero overlap with the human item universe. Therefore the drop is evidence that the human fine-tune substantially overwrites features useful to the much larger weak-labelled population.

v19 tests one narrow intervention against this measured failure. It does not replace or mutate the already queued v18 ladder.

## Intervention

Training order for a v19 probe:

1. historical/scaled weak phase using the retained weak preparation;
2. human phase, exactly 1.0 epoch;
3. score human holdout and item-disjoint weak holdout (`pre_refresh`);
4. fixed weak refresh on weak-train only:
   - `refresh_epochs = 0.05`;
   - `refresh_learning_rate = 2e-6`;
   - effective batch 32;
   - same soft weak targets and confidence weights;
   - no weak-holdout row may be used for refresh;
5. rescore the same human and weak holdouts (`post_refresh`).

The intervention adds training cost only. Production inference remains the same one-checkpoint RuBERT pair CrossEncoder at max_length 256.

## Promotion gates

Historical fold0 refresh is promoted to scaled confirmation only if all are true:

- weak-holdout Macro AP `post - pre > +0.005`;
- human Macro AP `post - pre >= -0.002`;
- worst human category with >=200 rows has AP delta >= -0.03;
- soft-target Brier score on weak holdout does not worsen by more than 0.002;
- sealed gold remains unopened;
- human/weak item overlap remains zero.

Scaled confirmation runs folds 0 and 1 at `3,000,000 -> 1,500,000 x 1.0` weak exposure and the same fixed refresh. Production is promoted only if:

- each fold independently passes the historical gates;
- mean human refresh delta across folds is >= 0;
- mean weak refresh delta across folds is > +0.005;
- gold remains unopened.

Thresholds are immutable after any v19 candidate metric is observed.

## Production

Full-development training uses the same scaled weak phase, all 285,210 development rows for the human phase, then the same `0.05 @ 2e-6` weak refresh over production weak-train. The final model is saved only after refresh and remains a single safetensors checkpoint. Packaging must pass the existing exact 1000-pair organizer Check.

## Relationship to v18

v19 is an independent anti-forgetting line. If both v18 strengthening mechanisms and v19 refresh later prove useful, the next integration experiment may combine them, but neither line is relabelled or post-hoc modified based on the other's results.
