# E-CUP 2026 Matching — solution research v2

Date: 2026-08-10

This document records the solution search performed before model training. It intentionally contains no raw product rows, product names, attributes, item IDs, or competition labels beyond aggregate statistics produced by the private/ephemeral profiling workflow.

## 1. What the task actually rewards

The task is pairwise product identity classification after candidate retrieval has already happened. The final score is macro Average Precision across 20 categories, so the model must rank true matches above non-matches inside every category rather than merely optimize global accuracy. The evaluation runs offline and has strict runtime limits, so latency is part of the model-selection problem rather than an afterthought.

The organizer baseline is a pairwise MiniLM Cross-Encoder. It is a useful reference but it repeats Transformer computation for every candidate pair, which is exactly the expensive operation we should try to avoid.

## 2. Measured human-data profile

The GitHub Actions job `profile-human` successfully downloaded and profiled the human-labelled subset on 2026-08-10. The generated artifact contains aggregate statistics only.

Observed facts:

- 365,654 human-labelled pairs.
- 711,304 unique referenced items; all referenced items are present.
- 20 categories.
- Overall positive rate: 25.6773%.
- All human pairs are within the same category; cross-category pair count is zero.
- Positive rate varies strongly by category: approximately 7.26% to 56.20%.
- Product names: mean 57.68 characters, median 54, p95 113.
- Pair name length: p50 108, p90 180, p95 212, p99 285 characters.
- Attributes: mean 442.30 characters, median 353, p95 1,070.
- 13.89% of item rows collapse onto an already-seen normalized name under the initial conservative normalizer.

### Consequences

1. **Global accuracy is misleading.** Some categories have far more positives than others, while the competition macro-averages category AP. Every experiment must report 20 per-category AP values plus macro AP.
2. **Category is effectively a blocking variable.** Since the human pairs are entirely intra-category, category-specific interactions and category-dependent decision functions are cheap and promising.
3. **Names are short enough for efficient encoders.** A 128-token name-centric semantic encoder should cover most names without aggressive truncation.
4. **Raw attributes are too long for indiscriminate pairwise attention.** Feeding two full attribute blobs through a Cross-Encoder for every pair wastes much of the runtime budget. Structured comparison should happen before any expensive pair model.
5. **Exact identity signals matter.** The normalized-name duplicate rate is non-trivial, but exact names cannot solve the task alone; they should be a high-precision feature rather than the whole model.

## 3. What public competitors / analogous systems do

As of the first day of the competition, public web/GitHub searches did not surface reproducible E-CUP 2026 matching solutions from current participants. The official FAQ also explicitly says solutions must be the participant's own and copying another participant's solution is not allowed. Therefore this research uses established public entity/product-matching systems as the competitive reference set rather than pretending we know private leaderboard methods.

### Ditto

Ditto serializes structured records with explicit column/value markers and fine-tunes a Transformer as a pair classifier. It also explores product-domain normalization, augmentation, and TF-IDF-style summarization. This validates the idea that explicit attribute structure and product identifiers/numbers should be preserved rather than flattening everything naively.

Reference: https://github.com/megagonlabs/ditto

### WDC Products benchmark

WDC Products evaluates Ditto, HierGAT, and supervised-contrastive methods under unseen-entity and corner-case conditions. Its authors report that all systems degrade on unseen entities and that contrastive learning is more training-data efficient than Cross-Encoder approaches. This is highly relevant because E-CUP test products are unseen during training.

References:

- https://webdatacommons.org/largescaleproductcorpus/wdc-products/
- https://arxiv.org/abs/2301.09521

### Supervised contrastive product matching

Peeters & Bizer show that supervised contrastive pre-training improves product matching on several benchmarks and is especially useful when explicit supervision exists. That supports learning an item-level semantic space before the final pair classifier.

Reference: https://arxiv.org/abs/2202.02098

### 2026 LLM-to-small-model distillation work

Recent 2026 entity-matching research investigates using an LLM only as a training-time teacher and a much smaller matcher at inference. This is directly aligned with E-CUP's huge LLM-labelled pool plus strict inference budget: exploit teacher/weak-label information offline, but never pay large-model cost on every test pair.

References:

- https://arxiv.org/abs/2606.28823
- https://arxiv.org/abs/2602.05452

## 4. Ten candidate solutions

Scoring is an engineering prior before leaderboard evidence. Scale 1–10. Weighted score uses:

- expected Macro AP: 40%
- inference speed: 25%
- unseen-item generalization: 15%
- ability to exploit weak labels: 10%
- implementation/risk: 10% (higher is safer/easier)

| # | Solution | AP | Speed | Unseen | Weak labels | Risk | Weighted | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Organizer MiniLM Cross-Encoder | 6.5 | 3.0 | 6.0 | 6.0 | 9.0 | 5.75 | Reference only |
| 2 | Larger multilingual Cross-Encoder | 8.5 | 1.5 | 7.5 | 8.0 | 6.0 | 6.20 | Too slow |
| 3 | Tiny Cross-Encoder only | 6.8 | 8.5 | 6.2 | 7.0 | 8.5 | 7.33 | Fast baseline |
| 4 | TF-IDF/char n-gram linear model | 5.8 | 10.0 | 6.5 | 5.0 | 10.0 | 7.30 | Excellent sanity baseline |
| 5 | Attribute-aware CatBoost/LightGBM | 7.1 | 9.7 | 7.2 | 6.5 | 9.0 | 7.90 | Strong CPU baseline |
| 6 | Contrastive bi-encoder only | 7.7 | 9.0 | 8.7 | 7.5 | 7.2 | 8.04 | Strong semantic baseline |
| 7 | Bi-encoder + lexical/attribute GBDT | 8.4 | 8.8 | 8.7 | 7.8 | 7.5 | 8.43 | Very strong Pareto point |
| 8 | Cheap model + uncertainty Cross-Encoder cascade | 8.8 | 7.7 | 8.5 | 8.2 | 6.5 | 8.29 | Likely quality boost |
| 9 | Teacher-student distillation only | 8.6 | 8.2 | 8.5 | 9.5 | 6.0 | 8.37 | Strong but needs careful teacher labels |
| 10 | **Noise-aware distilled hybrid cascade** | **9.3** | **8.0** | **9.0** | **9.5** | **6.2** | **8.70** | **Selected** |

## 5. Selected solution: noise-aware distilled hybrid cascade

The selected solution is not a giant ensemble. It is one inference pipeline with a cheap universal path and a compact expensive path that is only used when necessary.

### Stage A — deterministic item normalization

For every item, precompute:

- Unicode/case/whitespace normalized name;
- tokens preserving Cyrillic, Latin, digits, punctuation that matters for models/SKUs;
- canonical units and numeric values (`128 GB`, `128GB`, `0.128 TB` where safe to convert);
- alphanumeric model identifiers (`rtx4070`, `sm-s921b`, `abc-123`);
- normalized attribute keys and values;
- selected high-information attributes by category;
- category ID.

Never strip digits or aggressively stem product names. Model numbers, dimensions, pack sizes, memory, color codes, and revision identifiers are often the decisive identity signal.

### Stage B — item-disjoint validation

Random pair splitting is forbidden for model selection because the test contains unseen products.

Build an undirected graph over human pair item IDs. Connected components are indivisible groups. Assign whole components to train/validation, stratifying approximately by category and positive rate. Assert that the item-ID intersection between train and validation is empty.

Primary metric:

```text
macro_ap = mean(average_precision_score(y_cat, score_cat) for cat in 20_categories)
```

Also maintain a `hard_validation` subset containing:

- high lexical similarity negatives;
- low lexical similarity positives;
- model/SKU conflicts;
- numeric/unit conflicts;
- same normalized name but negative label;
- human/LLM disagreement when weak labels are added.

### Stage C — cheap structured/lexical pair model

Compute vectorized pair features:

**Name features**

- char 3/4/5-gram cosine;
- word/token Jaccard;
- containment both directions;
- normalized edit ratio;
- exact normalized-name equality;
- prefix/suffix overlap;
- token-count and length ratios.

**Model/SKU features**

- exact model-token match;
- intersection count;
- conflict count;
- rare alphanumeric-token overlap;
- one-side-only model-token flag.

**Numeric/unit features**

- numeric-value intersection;
- conflicting numeric values under same unit/attribute key;
- dimension similarity;
- quantity/pack-size similarity;
- memory/storage similarity;
- unit-normalized absolute/relative differences.

**Attribute features**

- attribute-key Jaccard;
- shared-key count;
- exact value agreement rate over shared keys;
- normalized value agreement rate;
- contradiction count;
- category-specific key agreements;
- attribute coverage ratio.

Train CatBoost or LightGBM. Do not optimize a binary threshold; select by macro AP.

This model is the first serious baseline and also remains part of the final pipeline.

### Stage D — contrastive bi-encoder

Serialize each item once, for example:

```text
[NAME] ... [ATTR] brand=... ; model=... ; memory=... ; size=...
```

The attribute serializer should select/shorten attributes so the semantic encoder sees informative content rather than the complete 1,000+ character tail.

Train an open-license multilingual/Russian item encoder with supervised contrastive learning. Starting candidates:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — higher-quality starting point;
- `cointegrated/rubert-tiny2` — aggressive speed candidate.

Each unique evaluation item is encoded once. Pair features then use cosine, L1/L2 summaries, elementwise product summaries, and optional learned projection similarity.

### Stage E — weak-label curriculum over >11M LLM pairs

Do not concatenate the 11M LLM rows with the human set using equal weight.

Start with a source-aware curriculum:

- human rows: weight 10;
- LLM target `[0, 0.03]` or `[0.97, 1]`: weight 1.0;
- LLM target `(0.03, 0.15)` or `(0.85, 0.97)`: weight 0.6;
- LLM target `(0.15, 0.30)` or `(0.70, 0.85)`: weight 0.3;
- LLM target `(0.30, 0.70)`: initially exclude, later use for hard-example selection rather than direct truth.

Sweep these weights on the fixed item-disjoint validation set. The numbers are starting hypotheses, not sacred constants.

### Stage F — hard-negative mining

After the first hybrid model:

1. score a large sample of training/LLM pairs;
2. collect high-scoring negatives and low-scoring positives;
3. oversample conflicts involving model identifiers, numeric attributes, sizes, colors, pack quantities and revisions;
4. fine-tune the encoder/student on these errors;
5. preserve random/easy samples so training does not collapse onto pathological cases.

### Stage G — distilled compact Cross-Encoder

Use a stronger open-license reranker only offline as teacher on strategically selected pairs, especially uncertainty/hard-negative regions. Distill teacher logits plus gold human labels into a small student Cross-Encoder.

The student sees compact, structured text rather than raw full attributes. A tiny RuBERT/MiniLM-class model is preferred over a 0.6B+ model for submission inference.

Loss sketch:

```text
L = w_gold * BCE(student, human_target)
  + w_soft * KL/BCE(student, teacher_or_llm_soft_target)
  + optional ranking loss inside category
```

Gold human supervision always dominates weak supervision.

### Stage H — uncertainty cascade

Every pair first receives the cheap hybrid score. Run the compact Cross-Encoder only when the pair is likely to affect ranking materially.

Gate features include:

- cheap-model score percentile within category;
- disagreement between lexical/GBDT and bi-encoder scores;
- model-token/numeric contradictions;
- distance to empirically difficult score regions;
- hard-negative detector output.

Tune the cascade fraction over `{5%, 10%, 15%, 20%, 30%, 40%}` and record both macro AP and end-to-end wall-clock time. Choose the Pareto point with at least 25% safety margin to the competition limit.

### Stage I — category-aware refinement

Because positive rates differ dramatically and macro AP weights all categories equally, include category as a feature and evaluate lightweight category-specific heads/residuals. Do not immediately train 20 completely separate Transformers; only split capacity when per-category ablations prove a benefit.

## 6. Ten experimental iterations

### Iteration 1 — reproducible validation + exact features

Deliverables:

- item-disjoint split;
- macro/per-category AP implementation;
- feature cache;
- exact-name/model/numeric/attribute baseline;
- measured inference time.

### Iteration 2 — char/word retrieval-style features

Add TF-IDF/char n-gram similarities and compare against Iteration 1. This sets a strong no-neural lower bound.

### Iteration 3 — off-the-shelf bi-encoder

Generate cached item embeddings without fine-tuning. Add embedding similarities to the GBDT. This isolates the value of generic semantics.

### Iteration 4 — supervised contrastive fine-tuning on human data

Train only on human labels/components. Compare unseen-item validation AP to Iteration 3.

### Iteration 5 — weak-label curriculum

Add high-confidence LLM rows using source-aware weights. Sweep confidence ranges and human:LLM weight ratios.

### Iteration 6 — hard-negative mining

Mine failures from Iteration 5, retrain and measure especially low-AP categories and the hard-validation slice.

### Iteration 7 — compact Cross-Encoder student

Train a small pair model on gold + selected weak/teacher examples. Benchmark full-pair inference even though it will probably not be the final deployment mode.

### Iteration 8 — uncertainty cascade

Invoke the student only for hard pairs. Sweep cascade fraction and gating logic.

### Iteration 9 — category-aware residuals

Find categories that remain materially below macro performance and train lightweight specialized residual models/heads only for them.

### Iteration 10 — final distillation/ensemble pruning

Use the strongest open-license offline teacher on informative examples, retrain the student, then remove any component that does not improve the AP/runtime Pareto frontier. Freeze the exact submission environment and measure Check/Public/Private-style workloads repeatedly.

## 7. Expected strongest route

The most likely winning development sequence is:

```text
structured GBDT
    ↓
+ cached bi-encoder similarities
    ↓
+ contrastive fine-tuning
    ↓
+ noise-aware LLM supervision
    ↓
+ hard-negative mining
    ↓
+ tiny distilled Cross-Encoder on uncertain pairs only
    ↓
+ category-specific residuals where justified
```

The key bet is that the competition is not won by the largest pair model. It is won by spending compute where pairwise interaction is actually necessary while exploiting exact product structure everywhere else.

## 8. Acceptance criteria for every new component

A component is kept only if all are true:

1. It improves fixed item-disjoint macro AP or substantially improves the weakest categories/hard slice.
2. The improvement reproduces over multiple seeds/splits where applicable.
3. End-to-end inference remains comfortably within runtime limits.
4. The model/package fits archive and Docker constraints.
5. The dependency/model license is allowed by the competition.
6. It does not depend on network access at inference.
7. It does not introduce validation leakage through repeated item IDs.

If a component improves public leaderboard score but fails the item-disjoint validation repeatedly, treat it as suspicious rather than automatically accepting it.
