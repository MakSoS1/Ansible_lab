# E-CUP Matching — Iteration v1 Results

Date: 2026-08-10

## Purpose

Iteration v1 is the first end-to-end, organizer-compatible submission anchor. Its goal is to establish a leakage-resistant validation score, measured runtime, exact submission contract compatibility, and a reproducible training/packaging path before adding weak labels and neural components.

## Model

`v1-structured-hgb`

- deterministic normalization of title and flattened JSON attributes;
- pairwise lexical/token/character similarities;
- exact and containment signals;
- numeric/model-code agreement/conflict;
- normalized quantity/unit agreement/conflict;
- attribute key/value agreement/conflict;
- category one-hot encoding;
- `sklearn.ensemble.HistGradientBoostingClassifier`;
- inverse category-frequency sample weighting;
- seed 2026.

The originally planned CatBoost/RapidFuzz v1 was replaced after probing the exact organizer image showed those packages are absent. Training and serialization were therefore executed inside `odsai/ecup26-matching-baseline:1.0` itself (Python 3.12.3, sklearn 1.9.0, joblib 1.4.2).

## Validation protocol

- Human labels only: 365,654 pairs total.
- Split: connected-component/item-disjoint.
- Train rows: 292,523.
- Validation rows: 73,131.
- Actual validation fraction: 0.20000055.
- Item IDs shared between train and validation: **0**.
- Metric: unweighted mean of sklearn `average_precision_score` over all 20 categories.

## Validation result

**Macro Average Precision: 0.4961654895**

| Category | AP |
|---|---:|
| Автотовары | 0.485144 |
| Аптека | 0.452134 |
| Бытовая техника | 0.662822 |
| Бытовая химия | 0.680373 |
| Галантерея и аксессуары | 0.345242 |
| Детские товары | 0.748140 |
| Дом и сад | 0.505511 |
| Канцелярские товары | 0.542651 |
| Красота и гигиена | 0.572134 |
| Мебель | 0.356299 |
| Музыкальные инструменты | 0.631460 |
| Обувь | 0.259856 |
| Одежда | 0.270446 |
| Продукты питания | 0.543658 |
| Спорт и отдых | 0.460734 |
| Строительство и ремонт | 0.499203 |
| Товары для животных | 0.593944 |
| Хобби и творчество | 0.784129 |
| Электроника | 0.216559 |
| Ювелирные изделия | 0.312870 |

## Training/runtime measurements

Measured inside the exact organizer image on a GitHub-hosted CPU runner:

- feature construction over train+validation: 285.20 s;
- HGB fit: 16.91 s;
- total train/evaluate pipeline: 308.57 s;
- packaged offline smoke: 1,000 pairs / 1,986 items;
- smoke load: 0.96 s;
- smoke feature construction: 0.79 s;
- smoke prediction: 0.03 s;
- smoke total: 1.78 s.

The 1,000-pair smoke is not a substitute for a full private-test benchmark, but it proves the packaged archive executes offline in the exact organizer image and provides a first throughput estimate.

## Submission artifact

Generated ZIP size: **581,818 bytes**.

Archive root contains the organizer contract files:

- `metadata.json` with image `odsai/ecup26-matching-baseline:1.0` and entry point `python -u run.py`;
- `run.py`;
- `model_v1.joblib`;
- `model_v1_manifest.json`;
- shared runtime feature modules under `ecup_matching/`.

The archive was executed with Docker `--network none`; output contained exactly `id1,id2,predict`, preserved pair order, had 1,000/1,000 rows, finite scores, and scores in `[0,1]`.

Private durable artifacts were verified at:

- `submissions/v1/model_v1.joblib`
- `submissions/v1/model_v1_manifest.json`
- `submissions/v1/v1-metrics.json`
- `submissions/v1/v1-metrics.md`
- `submissions/v1/ecup-v1-submission.zip`

inside private HF dataset `Maksim123321/e-cup-2026-matching-private`.

## What v1 teaches us

The model is already useful as a fast structured baseline, but the large category spread shows where semantic/product-specific modeling is needed. The weakest categories are Electronics, Shoes, Clothing, Jewelry, Accessories and Furniture. Those categories have many near-identical variants where exact model/size/color/style semantics matter, so generic lexical similarity is insufficient.

## Next iteration priority

Iteration v2 should add **filtered LLM weak labels + hard-negative mining** before adding a Transformer to inference. This is attractive because it can improve the same tiny runtime model while retaining essentially the full v1 speed. In parallel, prepare cached multilingual item embeddings for v3; then evaluate whether their Macro-AP gain justifies the additional archive/runtime cost.
