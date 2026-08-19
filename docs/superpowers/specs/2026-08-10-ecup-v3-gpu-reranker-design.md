# E-CUP v3 GPU Reranker Design

Date: 2026-08-10
Status: approved by the user's explicit instruction to proceed without stopping or asking additional questions

## Goal

Produce a retained v3 submission that improves the fixed leakage-resistant E-CUP 2026 validation score over v2b (`0.5010008995` Macro AP) by adding a compact neural pair reranker trained on model-mined hard negatives, while preserving a safe offline organizer runtime and keeping all competition data, weights, submissions and credentials private.

## Fixed constraints

- Work only on branch `ecup-matching-2026`; never modify/merge `main` unless explicitly requested.
- Keep the exact v1/v2 item-disjoint human validation: 73,131 pairs, zero overlapping item IDs.
- Official metric is unweighted category Macro Average Precision over all 20 categories.
- v2b structured model remains the always-available anchor and fallback.
- Public Git contains only code, aggregate metrics and source-backed documentation. Raw parquet, derived private text datasets, model weights and submission ZIPs stay in private HF `Maksim123321/e-cup-2026-matching-private`.
- No credential may be written to Git, logs, Memora, artifacts or model manifests.
- A retained v3 must pass organizer-image offline execution with network disabled and preserve comfortable headroom against the 780 s private limit.

## GPU backend strategy

### Preferred: Hugging Face ZeroGPU

Use a private Gradio ZeroGPU Space under the existing HF account. Current HF documentation states that free personal accounts in good standing can host up to two ZeroGPU Spaces and receive a small daily GPU quota. This route is preferred because the private training artifacts already live on HF and the existing GitHub `HF_TOKEN` secret can manage private Hub artifacts without exposing a new third-party credential.

The ZeroGPU call must perform GPU-only work. GitHub Actions CPU prepares a compact derived train/validation parquet first, so the five-minute GPU quota is spent on tokenization/training/scoring rather than scanning the 4+ GB source dataset.

### Fallback 1: Kaggle free GPU

Kaggle officially provides free P100-class GPU quota and its CLI can push/run kernels. Use it only if a Kaggle API credential already becomes available through a secure secret path; do not place competition data in a public Kaggle Dataset or notebook. Because no Kaggle credential is currently available to this agent, this is a fallback rather than the first implementation.

### Fallback 2: credit-backed providers

Do not consume paid compute. Modal/HF Jobs may be considered only when a clearly free/promotional credit balance can be established without charging a card. The user's request is for free GPU compute.

## Data design

1. Recreate the unchanged fixed human split.
2. Keep human outer-train labels authoritative.
3. Reuse the v2 weak-label curriculum and exclude every weak pair touching validation items.
4. Build pair text with the existing category-aware serializer (`reranker_data.py`).
5. To fit ZeroGPU time, create a stratified compact stage-1 set:
   - all human positives;
   - a balanced/stratified sample of human negatives, oversampling the six weakest v2 categories;
   - a confidence-filtered weak sample, again with extra mass on weak categories;
   - cap the first ZeroGPU attempt at a configurable row budget rather than silently reducing the validation set.
6. Keep the full 73,131-pair validation unchanged.

## Model and training

Base model: `cointegrated/rubert-tiny2` with a single sequence-classification logit.

Stage 1:
- symmetric pair serialization;
- max length initially 192 (256 remains an ablation if time allows);
- mixed precision on CUDA;
- weighted soft BCE;
- category-equalizing weights;
- one short fractional epoch sized to finish within ZeroGPU's call budget.

Stage 2 hard-negative mining:
- score authoritative human negatives with stage-1 reranker;
- take the highest-scoring false positives per category with explicit quotas for Electronics, Apparel, Footwear, Jewelry, Accessories and Furniture;
- pair them with positives from the same categories when possible;
- short low-learning-rate fine-tune;
- retain stage 2 only if Macro AP exceeds stage 1.

## Blend and routing

Evaluate four validation candidates:

1. neural reranker alone;
2. fixed global blends of v2 structured and neural scores (coarse sweep);
3. category-aware blend where neural weight is higher only in categories with validated gains;
4. uncertainty-gated reranking where v2 remains unchanged outside a bounded difficult subset.

Selection rules:
- choose only from the fixed validation;
- prefer the simplest candidate within 0.001 Macro AP of the best candidate;
- never accept a blend that reduces any category catastrophically without a compensating macro gain;
- keep v2 as fallback if neural training does not exceed `0.5010008995`.

## Submission architecture

The final v3 submit may not depend on network access. The archive contains:

- organizer-compatible `run.py` / `metadata.json`;
- retained compact structured model;
- retained compact reranker/tokenizer files only if runtime permits;
- deterministic pair serializer;
- blend/gating config.

Inference order:
1. load/normalize items once;
2. compute v2 structured score for all pairs;
3. select reranker subset if gating is retained, otherwise rerank all pairs;
4. run neural batches on organizer GPU;
5. blend scores;
6. write `id1,id2,predict` preserving pair order.

## Success criteria

A v3 is `completed` only if all are true:

- fixed validation overlap remains 0;
- selected Macro AP is strictly above `0.5010008995` (target >= `0.515`; stretch >= `0.53`);
- all 20 per-category AP values are recorded;
- model-mined hard negatives are actually used in a tested stage;
- final ZIP passes exact organizer image with network disabled;
- 275k-pair benchmark leaves at least 25% headroom to 780 s;
- private HF contains model/metrics/submission;
- `CURRENT`, `RESULTS`, experiment index, project state and durable decisions are updated;
- Memora policy/ingest/private checkpoint all pass.

If the neural result does not beat v2, record v3 as rejected/blocked rather than fabricating an improvement; continue to another safe free-GPU route only when it is actually automatable with available credentials.

## Testing

- Unit tests for compact sampler leakage/category balance and deterministic seed.
- Unit tests for model-mined hard-negative selection with per-category quotas.
- Unit tests for blend/gating selection and score range/order.
- Contract test that GPU worker never logs secret values and uploads only private artifacts.
- Existing repository tests remain green.
- Final organizer-image offline smoke and 275k benchmark are mandatory.
