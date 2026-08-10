# E-CUP 2026 Matching — Design

Date: 2026-08-10

## Goal

Build an isolated workspace for E-CUP 2026 Task 1 (product matching), mirror the organizer-provided parquet files into a private Hugging Face dataset repository, profile the data reproducibly on GitHub Actions, and develop a high-quality/low-latency matching approach optimized for Macro Average Precision and the competition runtime limits.

## Repository isolation

All new files live under `ecup_matching/`, `.github/workflows/ecup-matching.yml`, and `docs/superpowers/`. Existing Ansible files and other repositories are untouched. Work is performed on branch `ecup-matching-2026`.

## Competition constraints used in the design

- Candidate pairs are already retrieved; the submission only scores each provided pair.
- Training data contains 20 categories, ~365k human-labelled pairs and >11M LLM-labelled pairs.
- Human targets are binary; LLM targets are soft values in [0, 1].
- Test products are unseen during training.
- Metric: per-category `average_precision_score`, then macro-average across 20 categories.
- Runtime environment: 20 CPU cores, 200 GB RAM, H100 80 GB, no network.
- Runtime limits: Check 1 min, Public 6 min, Private 13 min.
- Submission archive <= 5 GB; Docker image <= 15 GB.
- The organizer baseline uses `cross-encoder/ms-marco-MiniLM-L12-v2` and is reported to take ~18.5 minutes across the full evaluation sequence, leaving almost no runtime margin.
- Speed is explicitly an important judging criterion in addition to leaderboard quality.

## Data mirror architecture

GitHub Actions downloads the four organizer files one at a time from the official Yandex Object Storage URLs and uploads each file into a private dataset repository `Maksim123321/e-cup-2026-matching-private`.

Files:

1. `matches.parquet` — human pair labels.
2. `matches_llm.parquet` — LLM soft pair labels.
3. `items.parquet` — full item corpus.
4. `items_human.parquet` — item subset referenced by human-labelled pairs.

The upload job:

- refuses to run if `HF_TOKEN` is absent;
- creates the destination repository as `private=True` and `repo_type="dataset"`;
- streams one source file to runner disk, checks non-zero size, uploads it, verifies it appears on the Hub, and removes the local copy before continuing;
- never commits raw competition data into Git;
- uses a write-capable Hugging Face token only through GitHub Actions Secrets.

A second Actions job profiles the smaller human-labelled data immediately and uploads only aggregate statistics as a workflow artifact. A full-data profiling path can be added after the first mirror succeeds.

## Validation strategy

The core validation rule is **item-disjoint splitting**, not a random pair split. The test set contains new items, so random pair splitting would leak item identity and overestimate generalization.

Recommended validation:

1. Build an undirected graph over item IDs appearing in human pairs.
2. Split connected components or item groups into train/validation so no item ID is shared.
3. Preserve category proportions as closely as possible.
4. Report AP separately for all 20 categories and their macro mean.
5. Keep a second "hard-only" validation slice containing pairs with high lexical similarity but opposite labels, numeric/model conflicts, and weak-label disagreement.
6. Use the same split for every ablation.

## Ten candidate solution families

### 1. Organizer Cross-Encoder baseline

Serialize both products and run MiniLM-L12 pair classification. Strong semantic interaction but already close to the time limit. Useful only as a reference and teacher candidate.

### 2. Larger Cross-Encoder

Fine-tune a stronger multilingual reranker/encoder directly on pairs. Potentially best raw pairwise accuracy but unacceptable inference cost for every test pair. Rejected as the final single-stage model.

### 3. Tiny Cross-Encoder only

Fine-tune a very small Russian/multilingual encoder such as a tiny RuBERT-derived classifier. Excellent latency and easy deployment, but it throws away the opportunity to exploit item-level reuse and structured attributes.

### 4. Pure TF-IDF / character n-gram model

Character and word n-gram cosine/Jaccard features plus Logistic Regression or LightGBM. Extremely fast and surprisingly strong on SKU/model-heavy names, but weak on paraphrases and semantically equivalent attributes.

### 5. Attribute-aware GBDT

Parse the JSON-like `attributes` field and train CatBoost/LightGBM on name overlap, attribute-key overlap, exact value matches, numeric conflicts, units, brands, model tokens and category. Very fast and interpretable; quality depends on normalization and cannot fully resolve semantic paraphrases.

### 6. Bi-Encoder cosine model

Fine-tune a multilingual sentence encoder contrastively, encode every unique test item once, and score pair cosine similarity. This removes repeated pairwise Transformer work and should be much faster than a Cross-Encoder. It is weaker on fine-grained "almost the same product" negatives.

### 7. Bi-Encoder + lexical/attribute GBDT

Use item embeddings once, then combine cosine/distance features with lexical, numeric and structured-attribute similarities in a small GBDT. Strong quality/speed ratio and highly suitable for unseen items.

### 8. Two-stage uncertainty cascade

Run the cheap model on every pair; invoke a small Cross-Encoder only for a narrow uncertainty/hard-negative band. Blend both scores. This preserves most of the Cross-Encoder benefit while spending expensive compute only where it changes ranking.

### 9. Teacher-student distillation from all labels

Use a stronger Apache/MIT-licensed teacher offline to generate logits on selected weak/hard pairs, then distill into a tiny student. Human labels receive much larger loss weight than LLM labels. This explicitly converts the 11M soft labels into useful scale without forcing a large model into the submission.

### 10. Category-aware mixture of experts

Train category-specific heads or a small set of experts because Macro AP weights every category equally. It can improve weak categories but increases training complexity and may overfit categories with little human supervision.

## Chosen architecture: noise-aware distilled hybrid cascade

The recommended final family combines ideas 5, 6, 8, 9 and a lightweight form of 10.

### Stage A — canonical item representation

For each product create deterministic representations:

- normalized `name` preserving model numbers, dimensions, quantities and units;
- canonical attribute string with stable key ordering;
- extracted numeric tokens and units;
- high-information alphanumeric model/SKU tokens;
- category ID.

Do not aggressively stem/remove digits: product identity often lives in model numbers and dimensions.

### Stage B — weak-label denoising

Human data is gold. LLM labels are auxiliary soft supervision.

Training examples get source-aware weights. Initial sweep:

- human: weight 8–12;
- LLM target <= 0.05 or >= 0.95: weight 1.0;
- LLM target in [0.2, 0.8]: weight 0.2–0.5;
- contradictory examples discovered by a teacher/heuristics: down-weight or move to a hard-case audit set.

The exact weights are selected by item-disjoint macro AP, never by random-split accuracy.

### Stage C — semantic item encoder

Start with an Apache/MIT-licensed multilingual/Russian encoder. Two practical starting points are `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for quality and `cointegrated/rubert-tiny2` for speed. Train with supervised contrastive objectives and hard negatives.

At inference, encode each unique item once. For each candidate pair compute cosine, elementwise-distance summaries, and optionally a compact learned bilinear score.

### Stage D — deterministic pair features

Fast CPU/vectorized features:

- name char 3–5 gram cosine;
- token Jaccard/containment;
- normalized edit similarity;
- exact normalized-name match;
- model-token exact/intersection/conflict indicators;
- numeric-token intersection and contradiction counts;
- unit-aware value similarity;
- attribute key Jaccard;
- exact/normalized attribute-value agreement ratios;
- embedding cosine/distance;
- category and per-category feature interactions.

A CatBoost/LightGBM model combines these features. Because AP is a ranking metric, optimize and select primarily on category-wise AP rather than threshold accuracy.

### Stage E — hard-pair Cross-Encoder cascade

Train a compact pair classifier by distilling a stronger teacher. Candidate student: `cointegrated/rubert-tiny2` or a custom 4–6 layer multilingual MiniLM student.

The cheap Stage-D model scores all pairs. Only the hardest ~10–30% are sent to the student Cross-Encoder. The gate is learned/tuned on validation using uncertainty plus disagreement features, not a fixed `0.5` threshold. Scores are blended monotonically within category.

A full Qwen3 reranker is appropriate as an offline teacher but not as the submission-time scorer. `Qwen/Qwen3-Reranker-0.6B` has an Apache-2.0 license and multilingual ranking capability, but the final student should be much smaller.

### Stage F — category-aware training and calibration

Use one shared feature/model backbone first. Add category-aware features and, only if validation proves it useful, category-specific lightweight heads. Since AP within each category is invariant to monotonic score transformations, focus on ranking quality rather than probability calibration.

## Why this should beat a full Cross-Encoder baseline

1. Reuses semantic computation per unique item instead of per pair.
2. Preserves product-critical exact/numeric/attribute signals that generic text rerankers can miss.
3. Uses 11M weak labels without treating them as equally trustworthy as 365k human labels.
4. Concentrates expensive pairwise attention on hard pairs only.
5. Distillation allows a strong teacher to influence inference without carrying its runtime cost.
6. Validation mimics unseen test items and Macro AP, reducing public-LB overfitting.

## Iteration ladder

1. Reproduce organizer baseline and measure local runtime/AP.
2. Build item-disjoint validation and deterministic text normalization.
3. Train lexical/attribute GBDT.
4. Add off-the-shelf bi-encoder embeddings.
5. Fine-tune bi-encoder on human labels.
6. Add high-confidence LLM soft labels with source-aware weighting.
7. Mine hard negatives and retrain.
8. Train compact Cross-Encoder student.
9. Add uncertainty cascade and optimize cascade fraction versus AP/runtime.
10. Distill a stronger Apache-2.0 teacher and ensemble only if the runtime budget remains comfortably below the limits.

## Success criteria

- Private HF mirror contains all four parquet files and remains private.
- Human-data profile artifact is generated reproducibly by GitHub Actions.
- Validation contains no item overlap between train and validation.
- Every experiment reports macro AP plus per-category AP and wall-clock inference time.
- Final submission has at least 25% runtime headroom versus the hard limits; target is substantially faster than the 18.5-minute organizer baseline.
- No final-model addition is accepted unless it improves the quality/runtime Pareto frontier.
