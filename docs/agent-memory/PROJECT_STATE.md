# E-CUP Matching — Canonical Project State

Updated: 2026-08-11

## Objective

Score strongly in the ODS E-CUP 2026 Ozon pairwise product matching task while keeping the final submission reproducible, legal under competition rules, robust to unseen items, and compatible with the organizer's offline container/runtime constraints.

## Official task constraints already established

- 20 product categories.
- Metric: unweighted macro mean of `sklearn.metrics.average_precision_score` across categories.
- Hidden test uses different/new products, so same-item leakage in validation is unacceptable.
- Training data: 365,654 human-labelled pairs plus >11M soft LLM-labelled pairs.
- Product universe: >13M items in the full item file.
- Submission runs offline in organizer container; input CLI uses `--items_path`, `--matches_path`, `--output_path` and output CSV is `id1,id2,predict`.
- Exact organizer image: `odsai/ecup26-matching-baseline:1.0`.
- Corrected image probe verified CUDA-oriented `torch 2.10.0+cu128`, Transformers, tokenizers, safetensors, numpy, pandas, sklearn and joblib. The v3 runtime therefore ships model/tokenizer assets but does not vendor Python wheels.

## Data and artifact state

Private HF dataset: `Maksim123321/e-cup-2026-matching-private`.

Mirrored organizer files:

- `matches.parquet`
- `matches_llm.parquet`
- `items.parquet`
- `items_human.parquet`
- `baselines/matching-baseline-submit.zip`
- `baselines/matching-baseline-lightweight.zip`

Canonical retained submission:

- `submissions/v3/ecup-v3-submission.zip`
- `submissions/v3/v3-package-metrics.json`
- ZIP bytes: `109,185,253`
- SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`

Do not copy raw competition data, learned model weights or submission ZIPs into this public Git repository.

## Experiment progression

### v1 — completed / superseded

`v1-structured-hgb` on the fixed item-disjoint split:

- train: 292,523 human pairs;
- validation: 73,131;
- shared item IDs: 0;
- Macro AP: `0.49616548946964434`.

Private artifact: `submissions/v1/ecup-v1-submission.zip`.

### v2 — completed / superseded anchor

Selected model: `v2b-weak-curriculum`.

Key additions:

- canonical pair-label cleanup and positive-component consistency checks;
- category-aware attribute importance;
- brand/model/number/quantity agreement and contradiction signals;
- symmetric product-aware/fuzzy structured features;
- confidence-filtered LLM weak-label curriculum.

Ablations on the unchanged validation:

- v2a human + structured product features: `0.5006971263`;
- v2b + filtered weak labels: **`0.5010008994958702`**;
- v2c static hard-negative weighting: `0.4957263069` — rejected.

Verified v2 exact-image benchmark: 275,000 pairs / 537,300 items / network disabled / 334 s CPU wall.

v2b remains the fast structured anchor and fallback inside v3.

## v3 — completed / current best

Architecture: **v2b structured score + compact `cointegrated/rubert-tiny2` pairwise reranker**.

Fixed validation is unchanged:

- rows: `73,131`;
- train/validation item overlap: `0`.

### Free GPU backend

The actual retained neural training ran on a standard GitHub-hosted `macos-15` Apple M1 runner with verified PyTorch MPS acceleration.

Other probes were rejected as infrastructure blockers:

- Hugging Face ZeroGPU returned HTTP 402 before allocation;
- Lightning Studio creation returned HTTP 403;
- Lightning direct Docker Job API authenticated but returned HTTP 403 even for a tiny CPU `Job.run`.

### Prepared v3 neural data

Successful preparation run: `31434855373` / job `93606589743`.

Private prefix: `experiments/v3/prepared/13edb087498b`.

- authoritative-human train before compaction: 292,523;
- compact train: 180,000;
- validation: 73,131;
- validation overlap: 0;
- retained human positives: 77,515;
- weak rows: 0.

The 4.1 GiB full-item weak-label path was intentionally abandoned for this retained neural iteration after a hosted runner shutdown. v2b still contributes its already-validated weak-label knowledge through the structured anchor.

### Retained neural run

Production run: `31437623156` / job `93615189602`.

Private prefix: `experiments/v3/neural/2d31cb18a06e`.

- stage 1: 1,600 steps;
- stage 2: 300 steps;
- total experiment time: 1327.387913542 s on Apple M1 MPS;
- hard-negative miner scored 180,000 authoritative rows;
- selected 12,000 model-mined hard negatives;
- selected 12,000 positives;
- 8,400 hard negatives came from the six priority categories;
- stage-2 focused set: 24,000 rows.

Stage 2 was genuinely trained and evaluated but did not improve the fixed validation; the retained checkpoint is stage 1.

### Final blend and quality

Selected: **`stage1-global`** with:

- structured v2b weight: `0.55`;
- neural weight: `0.45`.

Result:

- v2b: `0.5010008994958702` Macro AP;
- v3: **`0.5254642645846543` Macro AP**;
- absolute delta: **`+0.024463365088784106`**;
- relative gain: about 4.88%.

This makes v3 the current best retained solution.

### Runtime/package verification

Independent exact-image 1k smoke `31440472151` / job `93623920970` verified real neural execution with `1,000 / 1,000` neural pairs, network disabled and valid output.

Canonical package run: **`31440971110` / job `93625406492`**, source SHA `de4141af04e33170777d2de56ae059ebe52bb806`.

Fresh canonical evidence:

- repository tests: **108 passed**;
- memory/documentation policy: PASS;
- exact organizer-image ZIP build: PASS;
- ZIP: 109,185,253 bytes / 21 files;
- SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`;
- offline `--network none` smoke: 10,000 pairs;
- actual neural pairs: **10,000 / 10,000**;
- items: 19,822;
- GitHub CPU wall: 103 s;
- feature time: 6.56 s;
- neural time: 89.55 s;
- runtime total: 101.63 s;
- output rows/order/range/finite checks: PASS;
- unique prediction values: 9,996;
- canonical private HF upload and presence verification: PASS;
- cleanup: PASS.

The GitHub Ubuntu runner has no NVIDIA driver, so the CPU smoke is a correctness/stress check rather than an H100 throughput estimate. The organizer image is CUDA-capable and the runtime selects CUDA when available. A full 275k all-neural CPU stress run (`31439648374`) is non-blocking supplemental evidence because its CPU throughput is not representative of the organizer H100.

### Important diagnostic fixed during v3

A priority-category sprint package exposed a routing bug: item categories had been normalized to lowercase while manifest category keys retained display case, producing `neural_pairs=0` for that diagnostic package.

That sprint was rejected. RED/GREEN tests were added; the runtime now canonicalizes category keys and retains explicit global-alpha routing. The canonical v3 package was rebuilt from the corrected source and verified with 10k/10k real neural execution.

Full details: `ecup_matching/experiments/v3/RESULTS.md`.

## Current solution direction

Current retained model is v3. Future work is optional, not required to make v3 usable:

1. keep v3 as the submission candidate;
2. use v2b as the fast fallback;
3. only start v4 when an item-disjoint ablation has a plausible path to beat 0.5254642646 without jeopardizing organizer H100 runtime;
4. promising v4 directions include a better weak-label neural curriculum, selective category specialization, better hard-negative curriculum, or distillation/pruning if leaderboard/runtime evidence justifies it.

## Persistent agent memory — operational

The hardened Memora integration is installed, packaged and CI-verified.

Security/runtime profile:

- pinned upstream: `bc64ff745a9b2c0e6245e0137654f041fba0c155`;
- resolved MCP: `1.29.0` (`mcp>=1,<2` enforced);
- local SQLite + TF-IDF only;
- LLM, external embedding APIs, auto-capture and Memora cloud storage disabled in the supported project profile;
- content/metadata/tag secret redaction is verified behaviorally;
- local memory directory `0700`, DB `0600`;
- SQLite integrity and second-layer secret scan are required before upload.

Public Markdown is canonical. Private mutable state is stored under `agent-memory/latest/` in the private HF dataset, with retained checkpoints under `agent-memory/checkpoints/`. Every retained experiment must update PLAN/RESULTS/index/state, pass `memory_policy.py`, be ingested into Memora and create a verified private checkpoint before it is considered fully closed.
