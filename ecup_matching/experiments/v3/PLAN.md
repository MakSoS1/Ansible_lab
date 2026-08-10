# E-CUP Matching — Iteration v3 Plan

Date: 2026-08-10
Status: in progress

## Hypothesis

A compact pairwise Transformer trained on the existing human labels + confidence-filtered LLM labels, followed by model-mined hard-negative fine-tuning, will improve the fixed item-disjoint Macro AP over v2b. The largest expected gains are in Electronics, Apparel, Footwear, Jewelry, Accessories and Furniture, where structured lexical features remain weakest.

## Fixed comparison baseline

- v2b `v2b-weak-curriculum`
- Macro AP: `0.5010008994958702`
- validation: exact existing item-disjoint split, 73,131 human pairs, zero item overlap with outer train
- organizer runtime anchor: 334 s on 275k pairs / 537,300 items in `odsai/ecup26-matching-baseline:1.0`, network disabled

## Data

Private HF repository: `Maksim123321/e-cup-2026-matching-private`.

Inputs:
- `items_human.parquet`
- `matches.parquet`
- `items.parquet`
- `matches_llm.parquet`
- retained v2 model/validation artifacts

Training example policy:
- preserve every human positive in outer train when row budget permits;
- stratified human-negative sampling;
- confidence-filtered weak labels using the v2 curriculum;
- oversample/allocate extra training capacity to the six weak v2 categories;
- never use any weak pair touching a validation item;
- validation is never sampled or changed.

Initial compact row budget is 180k–260k examples, chosen to fit the 8 GiB local
GPU. If measured GPU speed allows, increase within the same validation protocol.

## Model

Base: `cointegrated/rubert-tiny2` sequence classifier, one logit.

Stage 1:
- category-aware pair text from existing `reranker_data.py`;
- max length 192 initially;
- mixed precision;
- weighted soft BCE;
- category balancing;
- short/fractional epoch bounded by the free GPU time budget.

Stage 2:
- score authoritative human training negatives with stage-1 reranker;
- select highest-scoring false positives, with per-category quotas and explicit priority for weak categories;
- pair with human positives;
- short low-LR fine-tune;
- keep stage 2 only when fixed validation Macro AP improves over stage 1.

## Blend candidates

Evaluate on the same fixed validation:
1. reranker alone;
2. v2/neural global alpha sweep;
3. category-aware alpha sweep;
4. uncertainty/category gated reranking if it improves score and materially reduces organizer runtime.

Every alignment must be exact by `(id1,id2)` and validation target/category.

## GPU backend order

1. Home RTX 2060 SUPER through private `MakSoS1/gpu-dispatch`, using an exact
   public-branch SHA and the fixed offline Docker profile.
2. Private Hugging Face ZeroGPU Gradio Space using only free quota if the local
   runner is unavailable.
3. Kaggle free GPU only if a secure Kaggle credential is already available; no public dataset/notebook.
4. Another provider only when it offers clearly free/promotional credits and secure automation without a paid charge. Do not silently spend money.

## Primary metric

Unweighted mean of `sklearn.metrics.average_precision_score` over all 20 categories.

## Success criteria

Minimum retained v3:
- validation overlap = 0;
- selected Macro AP > `0.5010008994958702`;
- target >= 0.515; stretch >= 0.53;
- all 20 category AP values recorded;
- actual model-mined hard-negative stage executed and measured;
- final organizer-image offline smoke passes;
- 275k benchmark <= 585 s (>=25% headroom to 780 s);
- private HF model/metrics/submission verified;
- memory policy, Memora ingest and private v3 checkpoint verified.

## Abort / reject criteria

- Any validation item leakage: abort run.
- Neural selected score <= v2b: do not call it a v3 improvement; retry a safe free-GPU configuration/backend if available, otherwise mark rejected/blocked.
- Organizer runtime >585 s: require gating/distillation/optimization before completion.
- Any need to publish raw competition data or expose a credential: reject that backend.

## Expected private paths

- `experiments/v3/prepared/<commit>/...`
- `experiments/v3/neural/<run-id>/...`
- `submissions/v3/ecup-v3-submission.zip`
- `submissions/v3/v3-validation-metrics.json`
- `submissions/v3/v3-package-metrics.json`
