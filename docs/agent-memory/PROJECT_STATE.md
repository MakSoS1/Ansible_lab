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
- Organizer runtime image verified as `odsai/ecup26-matching-baseline:1.0` with Python 3.12.3, numpy 2.2.3, pandas 2.2.3, sklearn 1.9.0, joblib 1.4.2.
- CatBoost and RapidFuzz were not present in that image during the runtime probe.

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

Validation:

- total human pairs: 365,654;
- train: 292,523;
- validation: 73,131;
- item IDs shared train/validation: 0;
- Macro AP: **0.49616548946964434**.

Private submit: `submissions/v1/ecup-v1-submission.zip`.

v1 is superseded by v2b but remains the reproducible baseline. Full details: `ecup_matching/experiments/v1/RESULTS.md`.

## Completed experiment v2 — current best

Selected model: **`v2b-weak-curriculum`**.

v2 transferred general product-matching lessons from public E-CUP 2024-style work without copying participant code:

- canonical pair-label cleanup and positive-component consistency checks;
- category-aware attribute importance;
- brand/model/number/quantity agreement and contradiction signals;
- explicit structured hard-negative score;
- symmetric fuzzy pair features;
- confidence-filtered LLM weak-label curriculum.

Structured ablations on the unchanged item-disjoint validation:

- v1 anchor: `0.4961654895` Macro AP;
- v2a human + 2024-inspired structured features: **`0.5006971263`**;
- v2b + 300k confidence-filtered weak LLM labels: **`0.5010008995`** — selected;
- v2c static hard-negative sample-weight boost: `0.4957263069` — rejected.

Weak-label safety:

- weak source rows examined: 11,187,780;
- confidence-filtered presample: 450,000;
- final weak sample: 300,000;
- unique weak-sample items: 559,153;
- any weak pair touching a fixed-validation item was excluded before training.

Private v2 artifact:

- `submissions/v2/ecup-v2-submission.zip`;
- `submissions/v2/v2-package-metrics.json`.

Verified organizer-image runtime benchmark from run `31427285112` / job `93581880597`:

- 275,000 pairs;
- 537,300 items;
- offline network disabled;
- schema/order/range checks passed;
- **334 s wall runtime** versus 780 s private limit;
- **446 s / 57.18% headroom**;
- ZIP size 603,415 bytes.

Full v2 details: `ecup_matching/experiments/v2/RESULTS.md`.

## Lightning / neural state

A compact `cointegrated/rubert-tiny2` pairwise reranker, weighted soft-label BCE curriculum and model-mined hard-negative second stage are implemented at code-contract level.

Lightning SDK `2026.8.5` authentication and Teamspace discovery worked through the secure ephemeral RSA credential bridge. GPU training did not begin because the account exposed no reusable Studio and `create_cloud_space` returned HTTP 403. No neural metric was fabricated and no GPU credits were consumed by those failed orchestration attempts.

The neural path is therefore deferred to v3 until an accessible Studio exists.

## Current best solution direction

Long-term architecture remains the **noise-aware distilled hybrid cascade**:

1. v1 — structured/lexical HGB anchor — DONE.
2. v2 — product-aware structured features + filtered weak labels — DONE; current best.
3. v3 — model-mined hard negatives + compact neural reranker; optionally add multilingual bi-encoder embeddings if runtime/quality justify them.
4. v4+ — tune weak-label curriculum, category specialization and blend only where item-disjoint ablations prove value.
5. final stage — pruning/distillation/runtime headroom pass.

Priority weak categories from v2b:

- Электроника — 0.257319 AP;
- Одежда — 0.267955;
- Обувь — 0.277386;
- Ювелирные изделия — 0.325490;
- Галантерея и аксессуары — 0.329753;
- Мебель — 0.371254.

## Immediate next action

Start v3 only after creating `ecup_matching/experiments/v3/PLAN.md`. Preserve the exact item-disjoint human validation. Use model-mined false-positive hard negatives rather than static reweighting, run the compact reranker on GPU when an accessible Lightning Studio exists, and retain the v2b structured model as the always-available fast anchor.

## Persistent agent memory — operational

The hardened Memora integration is installed, packaged and CI-verified.

Security/runtime profile:

- pinned upstream: `bc64ff745a9b2c0e6245e0137654f041fba0c155`;
- resolved MCP: `1.29.0` (`mcp>=1,<2` enforced);
- local SQLite + TF-IDF only;
- LLM, external embedding APIs, auto-capture, Cloud Graph/Pages/Worker, and Memora S3/R2/D1 storage disabled in the supported project profile;
- content/metadata/tag secret redaction is verified behaviorally;
- local memory directory `0700`, DB `0600`;
- SQLite integrity and second-layer secret scan are required before upload.

The memory workflow reads the current iteration dynamically from `ecup_matching/experiments/CURRENT.json`; it is no longer hard-coded to v1.

A verified v2 checkpoint already exists in private HF under `agent-memory/checkpoints/` and mutable current state is stored at:

- `agent-memory/latest/memories.db`;
- `agent-memory/latest/manifest.json`.

Public Markdown remains canonical. Agents with HF/shell access restore the latest DB with `python scripts/memory_bootstrap.py`; MCP clients launch only through `bash scripts/memora_mcp.sh`. Every retained v2+ experiment must update PLAN/RESULTS/index/state, pass `memory_policy.py`, be ingested into Memora, and create a private checkpoint before it is considered complete.
