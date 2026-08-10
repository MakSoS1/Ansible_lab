# E-CUP Matching — Canonical Project State

Updated: 2026-08-10

## Objective

Win/score strongly in the ODS E-CUP 2026 Ozon pairwise product matching task while keeping the final submission reproducible, legal under competition rules, robust to unseen items, and comfortably inside organizer runtime limits.

## Official task constraints already established

- 20 product categories.
- Metric: unweighted macro mean of `sklearn.metrics.average_precision_score` across categories.
- Hidden test uses different/new products, so same-item leakage in validation is unacceptable.
- Training data: 365,654 human-labelled pairs plus >11M soft LLM-labelled pairs.
- Product universe: >13M items in full item file.
- Submission runs offline in organizer container; input CLI uses `--items_path`, `--matches_path`, `--output_path` and output CSV is `id1,id2,predict`.
- Organizer runtime image currently verified as `odsai/ecup26-matching-baseline:1.0` with Python 3.12.3, numpy 2.2.3, pandas 2.2.3, sklearn 1.9.0, joblib 1.4.2.
- CatBoost and RapidFuzz were not present in that image during v1 probe.

## Data/artifact state

Private HF dataset: `Maksim123321/e-cup-2026-matching-private`.

Mirrored organizer files:

- `matches.parquet`
- `matches_llm.parquet`
- `items.parquet`
- `items_human.parquet`
- `baselines/matching-baseline-submit.zip`
- `baselines/matching-baseline-lightweight.zip`

Do not copy raw competition data or generated models/submission ZIPs into this public Git repository.

## Completed experiment v1

Model: `v1-structured-hgb`.

Core features:

- deterministic title/attribute normalization;
- lexical/token/character similarity;
- exact/containment similarities;
- number/model-code/quantity agreement and conflict;
- attribute key/value agreement and conflict;
- category encoding;
- sklearn `HistGradientBoostingClassifier`;
- inverse category-frequency weighting.

Validation:

- total human pairs: 365,654;
- train: 292,523;
- validation: 73,131;
- item IDs shared train/validation: 0;
- Macro AP: **0.49616548946964434**;
- feature build: 285.20 s;
- fit: 16.91 s;
- complete train/evaluate: 308.57 s.

Packaged submit:

- private HF: `submissions/v1/ecup-v1-submission.zip`;
- ZIP size: 581,818 bytes;
- offline smoke in exact organizer image: 1,000 pairs / 1,986 items in 1.78 s;
- pair order/schema/range validated with network disabled.

Weakest v1 categories and therefore high-priority error-analysis targets:

- Электроника: 0.216559 AP
- Обувь: 0.259856
- Одежда: 0.270446
- Ювелирные изделия: 0.312870
- Галантерея и аксессуары: 0.345242
- Мебель: 0.356299

Full v1 details: `ecup_matching/experiments/v1/RESULTS.md`.

## Current best solution direction

Selected architecture from the 10-idea research: **noise-aware distilled hybrid cascade**.

Planned progression:

1. v1 — structured/lexical HGB anchor — DONE.
2. v2 — use `matches_llm.parquet` with confidence curriculum/filtering and hard-negative mining; first try to improve the same tiny runtime model.
3. v3 — multilingual bi-encoder embeddings, item encoded once, pair embedding features added to cheap model.
4. v4+ — tune weak-label curriculum and hard negatives.
5. compact Cross-Encoder/student distillation.
6. uncertainty cascade: cheap model on all pairs, neural reranker only on difficult subset.
7. category-aware residuals only if ablation proves value.
8. final pruning/distillation/runtime headroom pass.

## Immediate next action

Create `ecup_matching/experiments/v2/PLAN.md`, profile the full `matches_llm.parquet` target distribution and item coverage without leaking hidden/test information, design confidence buckets, and train a v2 structured model using weighted soft labels plus human labels. Preserve the existing item-disjoint human validation unchanged so v1/v2 scores are comparable.

## Memory integration state

Memora is being integrated as a local-only, hardened semantic memory layer. Public Markdown remains canonical. Durable SQLite checkpoints go to private HF under `agent-memory/`. See `docs/agent-memory/SECURITY.md` and root `AGENTS.md`.
