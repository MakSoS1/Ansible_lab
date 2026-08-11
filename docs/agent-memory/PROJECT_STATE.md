# E-CUP Matching — Canonical Project State

Updated: 2026-08-11

## Objective

Score strongly in the ODS E-CUP 2026 Ozon pairwise product matching task while keeping the final submission reproducible, legal under competition rules, robust to unseen items, and compatible with the organizer's offline runtime.

## Fixed task/evaluation facts

- 20 product categories.
- Metric: unweighted macro mean of `sklearn.metrics.average_precision_score` across categories.
- Hidden test uses different/new products; same-item leakage in offline validation is unacceptable.
- Human training data: 365,654 labelled pairs; soft LLM data: >11M pairs.
- Fixed validation: 73,131 human pairs with **0 item IDs shared with outer train**.
- Submission CLI: `--items_path`, `--matches_path`, `--output_path`; output CSV columns `id1,id2,predict`.
- Exact organizer image: `odsai/ecup26-matching-baseline:1.0`.
- Public Git stores code/docs only. Raw competition data, model weights and submission ZIPs remain private.

Private artifact dataset: `Maksim123321/e-cup-2026-matching-private`.

## Current retained solution — v4

**v4 is the current best retained submission candidate.**

Architecture:

- unchanged v2b structured model from v3;
- unchanged retained `cointegrated/rubert-tiny2` neural checkpoint from v3;
- v4 replaces the single global neural blend alpha with a shrinkage-regularized per-category alpha vector.

The v4 model-selection protocol is deliberately leakage-aware at the routing layer:

- frozen validation rows: `73,131`;
- train/validation item overlap: `0`;
- routing cross-fit groups: connected components of **all validation candidate edges**;
- number of validation components: `53,131`;
- folds: 5-fold `GroupKFold`;
- tested shrinkage priors: `250, 500, 1000, 2000, 4000, 8000`;
- selected prior: **`4000`**.

Quality:

- retained v3 Macro AP: `0.5254642645846543`;
- cross-fitted global blend: `0.526005894031544`;
- **v4 cross-fitted regularized category blend: `0.5276431099433088`**;
- honest OOF delta vs v3: **`+0.0021788453586544243`**;
- deployable full-data coefficient fit: `0.5284493942551521`.

The `0.5276431099433088` cross-fitted value is the retained headline quality number. The larger `0.5284493942551521` value is recorded only as the score of the final coefficients fitted on all labelled validation rows after the prior was chosen OOF.

Selection run: `31473553650`.
Freeze/runtime run: `31474888023`, job `93726203398`.

Immutable source evidence:

- v3 canonical ZIP SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`;
- frozen v3 validation predictions SHA-256: `4112aa2556cb683ffca27cd9bd16c00a7149bb7e3279d1f2a6abb2b20438d643`.

Canonical v4 package:

- private prefix: `submissions/v4/canonical/b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`;
- ZIP SHA-256: **`b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`**;
- ZIP bytes: `109,185,879`;
- convenience alias: `submissions/v4/ecup-v4-submission.zip`;
- metrics alias: `submissions/v4/v4-package-metrics.json`.

Runtime gate:

- exact organizer image;
- `--network none`;
- read-only submission/runtime inputs;
- 1,000 smoke pairs;
- **1,000 / 1,000 real neural pairs**;
- valid output schema/order/range/finite checks;
- 1,000 unique prediction values;
- private canonical upload/presence verification PASS.

The hosted smoke ran the organizer image on CPU because the hosted Ubuntu runner has no NVIDIA driver. The retained v3 runtime automatically selects CUDA when available in the organizer environment.

Full details: `ecup_matching/experiments/v4/RESULTS.md`.

## v3 — immutable fallback

Architecture: v2b structured score + compact `cointegrated/rubert-tiny2` pairwise reranker.

Selected blend: structured `0.55`, neural `0.45`.

- Macro AP: `0.5254642645846543`;
- fixed validation rows: `73,131`;
- overlap: `0`;
- canonical ZIP SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`;
- canonical package smoke: 10,000 / 10,000 real neural pairs, network disabled, output verified.

v3 remains the immutable fallback and also supplies the learned structured/neural weights used by v4.

## Earlier iterations

### v1 — superseded

Structured HGB on the fixed item-disjoint split: `0.49616548946964434` Macro AP.

### v2 — superseded structured anchor

Selected `v2b-weak-curriculum`: `0.5010008994958702` Macro AP.

v2 introduced product-aware structured features and confidence-filtered weak labels. Its model remains the fast structured anchor inside v3/v4.

## Stronger encoder branch developed during v4

The original v4 design implemented an additional pinned `ai-forever/ruBert-base` ladder:

- v4a complete human curriculum;
- v4b confidence-filtered weak continuation;
- v4c model-mined hard negatives with 50% ordinary replay.

This branch is **not** the retained v4 artifact and no ruBERT metric is attributed to v4.

A first home RTX 2060 SUPER production attempt terminated with exit `137` during a host-memory-heavy preparation phase before any validation metric existed. The public code was then hardened:

- >11M-row weak presampling now streams with PyArrow rather than loading the full table into pandas;
- CPU-heavy structured/curriculum preparation precedes BERT loading;
- weak serialized pair direction is regression-tested;
- the private dispatcher fail-contains v4 at 10 GiB RAM with no extra swap.

The WSL runner went offline after the original host shutdown. A separate GitHub Apple-Silicon/MPS diagnostic remains supplemental only. Future work may resume this stronger encoder as **v4.1/v5**, but it must not overwrite or reinterpret the retained v4 evidence.

## Current action

1. Use canonical v4 as the current submission candidate.
2. Keep immutable v3 as fallback.
3. Do not modify the canonical v4 ZIP in place; any new model/alpha change must create a new immutable experiment/artifact.
4. Treat ruBERT-base training as a future ablation and retain it only if a new honest item-disjoint evaluation beats v4.

## Persistent agent memory — operational

The hardened Memora integration remains installed, packaged and CI-verified.

- pinned upstream: `bc64ff745a9b2c0e6245e0137654f041fba0c155`;
- resolved MCP: `1.29.0` (`mcp>=1,<2`);
- local SQLite + TF-IDF only;
- LLM/external embeddings/graph/auto-capture disabled in the supported profile;
- public Markdown is canonical;
- private mutable memory lives under `agent-memory/latest/` with immutable checkpoints under `agent-memory/checkpoints/`.

Every retained experiment must update PLAN/RESULTS/index/state, pass `memory_policy.py`, be ingested into Memora and create a verified private checkpoint before it is considered fully closed.