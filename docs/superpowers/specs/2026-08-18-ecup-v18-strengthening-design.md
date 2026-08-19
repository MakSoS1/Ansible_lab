# E-CUP v18 Strengthening Design

## Goal

Build the next E-CUP matching submission by improving the parts of the retained RuBERT CrossEncoder training pipeline that still have untested headroom: weak-label quality, hard-example selection, training-time invariances, trainable capacity, validation breadth, and active-learning data selection. Keep the production inference shape to one RuBERT pair CrossEncoder at max_length=256 so runtime risk does not grow with training quality.

## Evidence that constrains the design

1. v7 -> v12 is the only material observed Public-LB jump and the material change was weak supervision. The historical v12-v15 path reused 600,000 selected weak rows at 0.35 weak epochs, approximately 210,000 weak examples seen.
2. Later residual, graph, late-interaction, MoE and setwise changes produced small or negative external movement. Therefore v18 does not replace the production inference family before exhausting training/data improvements.
3. Current `weak_labels.py` quantizes confidence into weights 1.0/0.6/0.3 and removes the entire 0.30-0.70 probability band. This throws away potentially useful ranking information from the 11M+ weak pool.
4. Current `MacroPairBatchSampler` balances category and hard class but not sample difficulty.
5. Current pair training has no explicit order-symmetry augmentation and no field-view augmentation.
6. Current retained configuration fine-tunes only the last 8 encoder layers. Fine-tuning all 12 does not change inference runtime, only training memory/cost.
7. Human fold-0 alone is not a reliable external selector. v18 must preserve the item-disjoint weak-holdout axis and add a second human fold before production promotion.

## Non-negotiable constraints

- Sealed gold remains unopened and unscored.
- Human/weak item overlap remains exactly zero for every probe and production refit.
- Immutable human split SHA stays `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Production inference remains one RuBERT sequence-classification checkpoint, max_length 256, organizer image `odsai/ecup26-matching-baseline:1.0`.
- New training mechanisms may increase training time but must not add a second model, graph pass, TF-IDF model, or online feature model to final inference.
- All candidate selection rules are fixed before candidate metrics are read.
- Weak targets remain soft probabilities; no blanket thresholding to 0/1 is introduced.
- Synthetic augmentation must be label-preserving or explicitly low-weight and audited.

## Architecture

### 1. Quality-aware weak supervision (Q1)

Replace coarse confidence bins for v18 only with a continuous confidence function based on distance from 0.5. Keep a small dead zone around 0.5, but retain medium-confidence examples outside it with low weight instead of dropping the complete 0.30-0.70 band.

For probability `p`, let `margin = abs(p - 0.5)`. With dead zone `d=0.05` and exponent `gamma=1.5`:

`weight = 0` when `margin <= d`; otherwise `max(0.05, ((margin-d)/(0.5-d))**gamma)` clipped to 1.

This preserves very confident labels at weight ~1 while using medium-confidence labels as ranking evidence rather than authoritative truth.

Weak curriculum has two phases with the same total optimizer-step budget as its control:

- high-confidence phase: margin >= 0.30;
- broad phase: all retained examples, confidence-weighted.

### 2. Online hard-example mining (Q2)

After the first weak phase, score a deterministic mining subset with the current candidate checkpoint. Compute disagreement `abs(prediction-target) * weak_weight`. Select the highest disagreement examples approximately balanced by category and hard class, and mix them into the second weak phase.

Targets are never replaced with teacher predictions. Model predictions are used only to choose which existing weak-labelled examples deserve more training attention.

### 3. Symmetry and safe field-view augmentation (Q3)

Add deterministic per-epoch training-time views without increasing optimizer steps:

- pair-order swap with probability 0.5;
- drop the `[RESIDUAL]` line with probability 0.15;
- drop the `[NUMERIC]` line with probability 0.05 only when `[MODEL]` or `[IDENTITY]` evidence remains.

Never drop `[NAME]`, `[BRAND]`, `[MODEL]`, `[IDENTITY]`, or `[CAT]`. Pair target and weak confidence remain unchanged. The augmentation is training-only and production serialization stays byte-for-byte v7-compatible.

### 4. Full-encoder fine-tuning (Q4)

Compare last-12-layer fine-tuning to retained last-8-layer fine-tuning at the same data exposure. Use physical batch 16 / effective batch 32 and gradient accumulation if required by the 8 GiB RTX 2060 Super. Lower learning rate to `8e-6` for 12-layer probes. Inference architecture remains identical.

### 5. EMA checkpoint smoothing (Q5)

Maintain an exponential moving average of trainable parameters during optimization with decay `0.999`. Evaluate both raw and EMA weights on the same probe; only the better validated state is eligible for a final checkpoint. EMA is a training-time mechanism and adds zero production inference cost.

## Candidate ladder

All Q1-Q5 single-mechanism probes first run at the historical exposure (`weak_final_rows=600000`, `weak_epochs=0.35`) on human fold 0 with the same weak holdout seed.

A single mechanism is eligible for the combination stage only when all are true:

- weak-holdout after-human delta versus the matching v18 control is strictly > +0.003;
- human fold-0 delta is >= -0.003;
- no category with at least 200 held rows loses more than 0.03 AP;
- no provenance or overlap invariant fails.

The combination candidate includes only mechanisms that independently pass. It is first rerun at historical exposure to detect harmful interactions. Combination promotion requires:

- weak-holdout delta > +0.005 versus control;
- human fold-0 delta >= -0.002;
- combined robust score greater than every included single mechanism.

Only then is the combination scaled to `weak_presample_rows=3000000`, `weak_final_rows=1500000`, `weak_epochs=1.0`.

## Improved validation

Every metrics file reports four axes:

1. human Macro AP on the held component-disjoint fold;
2. item-disjoint weak-holdout Macro AP after weak phase;
3. the same weak-holdout Macro AP after human phase;
4. per-category AP and worst-category regression on categories with >=200 rows.

A scaled candidate must also run human fold 1 using the same frozen split contract. Final production promotion requires:

- fold0 human delta >= -0.002 versus the matching control;
- fold1 human delta >= -0.002 versus the matching control;
- mean of fold0/fold1 human deltas >= 0;
- weak-holdout delta > +0.005;
- worst qualifying category regression >= -0.03 on both human folds;
- gold remains unopened.

The validation output is explicitly a promotion system, not a fabricated leaderboard-score estimator.

## Active-learning / data-diversity export

For every probe, write a bounded parquet/CSV manifest containing examples with highest uncertainty/disagreement, stratified by category and by positive/negative weak target. Include pair ids, category, current weak target, confidence weight, candidate prediction, disagreement, and deterministic reason code.

This manifest is an input to later human/open-source-LLM auditing if credentials/rules permit, but v18 training must not depend on new external labels. No private dataset or model checkpoint is published publicly.

## M1/MPS smoke lane

Use public `macos-15`/`macos-latest` GitHub-hosted ARM64 runners only for inexpensive correctness checks. GitHub currently documents standard macOS ARM64 runners as M1 with 7 GB RAM. The workflow must probe `torch.backends.mps.is_available()` rather than assume GPU exposure. If MPS exists, run a tiny in-memory model through the v18 training path on MPS; otherwise run the same test on CPU and record the limitation.

This lane checks sampler, continuous weighting, curriculum partitioning, hard-example selection, deterministic augmentations, EMA bookkeeping, and a tiny train step. It does not claim performance comparability with the RTX lane and does not consume the canonical private dataset.

## RTX executor

The private `gpu-dispatch` branch queues the canonical candidate ladder behind the existing v17 concurrency group. Each run checks out one exact public source SHA, mounts canonical data read-only, uses the trusted offline container, writes outputs to persistent runner storage, and attempts private Actions artifact upload without making artifact quota a correctness dependency.

Production packaging reuses the v7-compatible single-checkpoint builder and the organizer-shaped 1000-pair runtime gate. Final ZIP is emitted only after the pre-registered v18 promotion checks pass.

## Failure policy

- A failed mechanism is recorded and excluded; do not weaken thresholds after seeing its metrics.
- OOM in Q4 may reduce physical batch size while keeping effective batch 32 and all other hyperparameters fixed; this is an execution fix, not a metric-driven tuning change.
- If no single mechanism passes, v18 does not fabricate a combined keeper. The existing v17 path remains the submission candidate.
- If combination passes local validation but the organizer runtime gate fails, no archive is promoted.
