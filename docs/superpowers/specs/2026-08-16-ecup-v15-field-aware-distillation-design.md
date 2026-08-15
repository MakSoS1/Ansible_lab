# E-CUP v15 Field-Aware CrossEncoder + Offline Distillation — Design

**Date:** 2026-08-16

## Goal

Reach a materially higher Public-LB regime than the v12/v13 ~0.38 plateau while preserving a submission architecture that can satisfy the organizer's 1/6/13 minute Check/Public/Private limits. The target is Public Macro AP >= 0.50, treated as a research objective rather than a guaranteed local-to-LB conversion.

## Why v15 exists

The v7-v13 family established that one compact pair CrossEncoder is fast enough and materially stronger than the older structured/TF-IDF stack, but local fold gains do not reliably order close Public-LB candidates. v13 improved fold0 over v12 yet scored lower on Public. The v14 item-centric family is genuinely new, but A0/A1/A3 and later token-cross screens show a large quality deficit against the full pair CrossEncoder: independent item encoding loses too much pair-conditioned reasoning for a competition in which Retrieval is already supplied by the organizer.

Therefore v15 does not return to v12 unchanged. It keeps the strongest proven inductive bias — full pair-conditioned token interaction — and changes the information presented to the model, the attribute reasoning path, the supervision strategy, and the experiment discipline.

## Architecture contract

### 1. Field-aware pair serialization

Each product is parsed deterministically into explicit fields before tokenization:

- title;
- brand when recoverable from attributes;
- model/SKU-like tokens;
- normalized structured attributes;
- numeric attributes with normalized units when safely parseable;
- category.

The Transformer receives one pair sequence with explicit stable field markers. No runtime network calls or dynamic schemas are allowed.

Conceptual form:

```text
[CLS]
[A_TITLE] ...
[A_MODEL] ...
[A_ATTR] key = value ...
[SEP]
[B_TITLE] ...
[B_MODEL] ...
[B_ATTR] key = value ...
[SEP]
```

A/B ordering may be augmented during training, but the final score contract must be symmetric within a frozen numerical tolerance. The implementation may use symmetric score averaging or a commutative fusion head; the selected mechanism must be measured for runtime and quality.

### 2. Typed deterministic pair evidence

A compact vector of deterministic features is fused into the neural head, not executed as a separate model branch. Initial feature families:

- exact and normalized model-token agreement/conflict;
- SKU/alphanumeric overlap/conflict;
- numeric token overlap and conflict counts;
- capacity/size/quantity agreement/conflict where unambiguous;
- brand agreement/conflict;
- title token overlap;
- attribute-key overlap;
- attribute-value agreement/conflict;
- missingness and field-length diagnostics.

This reuses information proven useful by historical structured models without reintroducing the HGB/TF-IDF runtime architecture that timed out.

### 3. Category specialization

The expensive encoder remains shared. Category specialization is lightweight and runs after the pooled pair representation. The first implementation uses a small category-conditioned residual head/adaptor and must be compared against an otherwise identical shared-head control. Category specialization is accepted only if strict category-level evidence improves Macro AP rather than merely global BCE.

### 4. Macro-oriented training

The official metric is the unweighted mean of 20 category AP values. Training therefore records and controls category exposure explicitly. The ladder starts with category-balanced sampling and compares it against the inherited sampler. No change is kept unless the effect is causally isolated.

### 5. Offline teacher and active distillation

The historical `matches_llm.parquet` is treated first as an unlabelled retrieval candidate graph (`id1,id2`) rather than trusted ground truth. The old `target` column remains quarantined unless a separate audit proves it reliable.

A strong offline teacher is allowed to be substantially more expensive than the final model because it never ships in the submission. Teacher candidates may combine:

- a stronger/full-capacity field-aware pair model;
- a human-only pair teacher;
- structured deterministic evidence;
- optional setwise candidate context;
- only open-license models if additional model-generated supervision is used.

The teacher is evaluated on held human truth before it may label unlabelled retrieval candidates. Distillation data are selected actively rather than uniformly, prioritizing hard or informative pairs such as near-duplicate negatives, model/SKU conflicts, numeric conflicts, low-similarity positives, teacher/student disagreement, and high-degree retrieval anchors.

The final student remains one offline field-aware pair checkpoint.

## Label policy

1. Human labels are authoritative for validation and the clean architecture ladder.
2. Historical weak/LLM targets are not trusted by default.
3. Using the historical weak file as an unlabelled `id1,id2` candidate source is permitted.
4. Any new pseudo-label source must be generated reproducibly and fold-safely.
5. Teacher quality must be measured on held human truth before pseudo-label generation.
6. Held-fold and sealed-gold item identities must not leak into fold training or distillation pools.
7. Sealed gold remains unopened.

## Validation v5

The canonical human split remains immutable:

- 365,654 total human rows;
- 285,210 development rows;
- 80,444 sealed-gold rows;
- 5 component-disjoint development folds;
- split SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- canonical row-map SHA-256 `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`;
- cross-split item overlap must remain zero.

Fold0 is a screen only. A production candidate requires strict five-fold OOF over exactly 285,210 development rows.

Every retained experiment reports:

- Macro AP;
- AP for all 20 categories;
- delta vs the v12-compatible reference on identical held rows;
- median and worst category delta;
- retrieval-hard slices;
- model/SKU conflict slices;
- numeric/capacity/size conflict slices;
- accessory/main-product slice when derivable;
- missing/long/short attribute slices;
- deterministic bootstrap stability of delta;
- fit and inference runtime.

Public-LB scores are experiment-level external anchors only and are never converted into row labels or a fitted local-to-LB formula.

## Experiment ladder

### V15-A0 — field-aware serialization control

Start from the proven single pair-CrossEncoder runtime family. Change only input construction from raw pair text to deterministic field-aware serialization. Human labels only for the clean screen.

### V15-A1 — normalized structured attributes

Add deterministic JSON parsing, normalized key/value representation, model/SKU extraction and conservative numeric normalization. No typed side-vector yet.

### V15-A2 — typed pair-feature fusion

Fuse deterministic typed pair evidence into a small neural head while preserving one Transformer checkpoint and one inference pipeline.

### V15-A3 — category-conditioned residual head

Add lightweight category specialization after the pair representation. Compare against A2 under identical data exposure.

### V15-A4 — macro-oriented sampling/loss control

Compare category-balanced exposure and category-local ranking auxiliary loss against A3 without changing the representation.

### V15-B0 — strong human-only teacher

Train a teacher that must beat the selected A-family student on held human truth by a predeclared material margin before any pseudo-label generation.

### V15-B1 — active unlabelled retrieval selection

Use only `id1,id2` from the historical weak pool to build an informative candidate subset. Do not read legacy `target` for training labels.

### V15-B2 — fold-safe teacher scoring

Score the selected unlabelled candidate subset with the stronger teacher, recording teacher uncertainty and disagreement with deterministic evidence.

### V15-B3 — distilled student + human recovery

Train the selected A-family student on fold-safe teacher soft targets, then finish with a clean human recovery phase. Compare directly against its human-only parent.

### V15-C — optional open-model relabeling

Only if the teacher program remains supervision-limited, evaluate a reproducible open-license model on held human truth and admit new pseudo-labels only after explicit category/failure-slice reliability gates. This phase is optional and must not block the A/B ladder.

## Promotion and stop gates

Fold0 exists to reject weak ideas cheaply.

- `delta < +0.005` Macro AP vs identical v12-compatible fold0 reference: retain as research evidence; do not spend five-fold GPU budget unless hard-slice evidence reveals a clearly different failure mode worth studying.
- `+0.005 <= delta < +0.010`: inspect category and hard-slice stability before deciding on another fold.
- `delta >= +0.010`: strong candidate for strict five-fold OOF.

These are GPU-budget gates, not a Public-LB calibration.

A final keeper requires:

1. complete five-fold OOF over exactly 285,210 rows;
2. zero duplicate OOF row indexes;
3. zero train/held item overlap;
4. sealed gold unopened;
5. all 20 category APs recorded;
6. majority of categories non-negative vs the frozen reference;
7. no catastrophic category regression (`worst delta >= -0.03` unless an explicit category-size/label issue is proven);
8. bootstrap stability of positive Macro-AP delta;
9. organizer-shaped runtime and exact package checks;
10. exact source/model/package SHA provenance.

## Runtime contract

The final submission must contain one tokenizer, one field-aware pair Transformer checkpoint and only lightweight deterministic parsing/fusion code. No teacher, HGB, TF-IDF, graph engine, second Transformer, online lookup or network dependency ships in the archive.

Runtime engineering rules:

- materialize only referenced supplied items;
- avoid full-item-universe scans;
- parse attributes once per referenced item;
- cache normalized item fields within a run;
- dynamic padding and GPU batching;
- benchmark FP16/BF16 where numerically safe;
- separate startup, parsing, tokenization, model-forward and output-write timings;
- Check target <= 50 s internally for >=10 s headroom;
- Public and Private tests use organizer-shaped row counts and conservative headroom;
- exact final ZIP bytes are timed, not an approximation.

## Repository and memory architecture

`MakSoS1/Ansible_lab` is the single canonical source of research truth. It owns:

- experiment PLAN/RESULTS/SAFE_METRICS/MANIFEST;
- `CURRENT.json`;
- `docs/agent-memory/*`;
- public/reproducible source and tests;
- durable decisions and experiment index;
- Memora ingestion/checkpoint policy.

`MakSoS1/gpu-dispatch` is an execution plane. It owns:

- immutable job request files;
- workflow definitions;
- runner/private-data adapters;
- private execution logs/artifacts;
- minimal executor documentation that points back to the canonical public experiment contract.

Every GPU job must declare at least:

```json
{
  "experiment_id": "v15-a0",
  "architecture_family": "pair_crossencoder",
  "role": "screen",
  "public_source_sha": "<40-char SHA>",
  "architecture_contract_path": "ecup_matching/experiments/v15/PLAN.md",
  "split_sha256": "aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b",
  "canonical_rowmap_sha256": "00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f",
  "label_policy": "human-only",
  "fold": 0
}
```

Private executor notes may contain private paths/run IDs, but architectural decisions are not canonical there. After each KEEP/REJECT/FAIL, the canonical public memory is updated before the result is considered fully handed off.

## Morning deliverable objective

The overnight queue should maximize information, not blindly produce many archives. The desired morning state is:

1. A0/A1/A2 screens completed if GPU time permits;
2. at least one materially promising architecture selected or the ladder causally rejected;
3. runtime smoke evidence for the selected family;
4. teacher preparation started only if a student is strong enough to justify it;
5. no production submission built from a candidate that has not passed its quality gate;
6. if one or more candidates pass quality and runtime gates, immutable production/refit/package jobs are available to turn them into uploadable submissions without reconstructing provenance manually.

This prevents another cycle of packaging models that are locally ambiguous or architecturally weak merely to have a ZIP.