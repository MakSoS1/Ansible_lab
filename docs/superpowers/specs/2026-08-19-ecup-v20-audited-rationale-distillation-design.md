# E-CUP v20 — Audited Data-Centric Rationale Distillation

Date: 2026-08-19
Status: awaiting written-spec review; implementation not started
Branch: `ecup-v20-data-centric`

## 1. Objective

Build a materially stronger product-matching submission by improving supervision density and quality over the weak-labelled item universe while retaining the proven single-checkpoint `ai-forever/ruBert-base` pair CrossEncoder runtime shape.

v20 has three goals:

1. audit and stratify existing weak supervision instead of treating confidence margin as the main notion of quality;
2. generate additional informative real-item pairs and admit only labels whose quality is empirically measured on authoritative human data;
3. distill both binary decisions and structured matching reasons into the same runtime-efficient RuBERT student.

The final submission is self-contained and offline. LLMs are training/data-preparation tools only. The current E-CUP FAQ explicitly permits LLM use for relabelling in the product-matching task. No API call, paid key, remote model, or second inference service is permitted in the submitted runtime.

## 2. Evidence from v1–v19 that constrains the design

### 2.1 Keep the RuBERT pair CrossEncoder family

The strongest useful architectural jump came from neural pair modelling, and the later strong `ai-forever/ruBert-base` CrossEncoder family substantially exceeded compact/alternative backbones locally. Granite, late-interaction/token-cross, graph and small residual families did not establish a better production parent.

Therefore v20 does not replace the runtime backbone unless a bounded ablation proves a clearly superior checkpoint under the same runtime envelope. Baseline production shape: one RuBERT pair CrossEncoder, max length 256, one `.safetensors` checkpoint.

### 2.2 Human fold-0 is a safety axis, not the selector

Known external anchors:

| Candidate | Human fold-0 | Public LB |
|---|---:|---:|
| v7 | 0.7023802626 | 0.3655833314 |
| v12 | 0.7059297810 | 0.3798116204 |
| v13B | 0.7086611386 | 0.3783781653 |
| v14 | 0.7065769714 | 0.3803270470 |

Human fold-0 misranks v12/v13B/v14, so a tiny fold-0 gain cannot promote v20 by itself.

### 2.3 Weak supervision is under-used

The canonical weak corpus contains 11,187,780 pairs over 12,384,610 items. The historical v12+ recipe used `weak_final_rows=600000` and `weak_epochs=0.35`, roughly 210,000 examples seen, ~1.88% of the pool.

Human labels cover 711,304 items (~5.31% of 13,397,761 items); weak labels cover ~92.44%. The two supervision populations share zero item IDs, but their text/attribute density is similar.

### 2.4 Weak-label quality was never actually audited against human truth

The historical weak audit had zero overlap with human items, so positive/negative precision could not be estimated. v20 instead audits the exact candidate LLM labelling pipeline by re-labelling a held authoritative human calibration subset.

### 2.5 Hard-negative-only curricula are not trusted

Naive hard-negative weighting regressed in v2 and the model-mined hard-negative second stage was rejected in v3. v20 balances semantic reasons and preserves broad replay instead of concentrating training on the most disagreeing negatives.

### 2.6 v17 measured a forgetting-like shift, but weak truth is uncertain

The v17 control measured weak-holdout AP `0.6973930799` after weak training and `0.6565798751` after human fine-tuning, with human fold-0 `0.7017637364`. This is a real shift on the weak axis, but weak labels remain pseudo-truth. Anti-forgetting mechanisms are inherited only if they pass human and audited-stratum gates.

## 3. Immutable data contract

Sources:

- `items.parquet`: 13,397,761 item records;
- authoritative `matches.parquet`: 365,654 human-labelled pairs;
- historical `matches_llm.parquet`: 11,187,780 weak-labelled pairs;
- frozen five-fold component/item-disjoint split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- sealed gold: 80,444 rows, never opened or scored for model selection.

For outer fold `k`, every generated/audited/training pair excludes every item ID belonging to fold `k` before candidate generation, LLM labelling, teacher scoring, mining, or training.

Generated private layers:

- `bronze/audit_candidates.parquet` — human calibration + target-free candidate pairs;
- `silver/llm_labels.parquet` — raw structured teacher outputs and provenance;
- `silver/admitted_labels.parquet` — only statistically admitted labels;
- `gold/train_pairs_fold{k}.parquet` — fold-safe human + admitted weak/generated rows;
- `gold/active_review.parquet` — uncertainty/disagreement rows excluded from training.

Every generated row carries `source`, `generator_version`, `stratum`, `reason_code`, `label_origin`, teacher IDs/revisions, prompt hash, admission-policy hash and fold-exclusion evidence.

## 4. Semantic data audit

D1 scans the complete existing weak corpus and relevant item fields without fitting a model. For each official category and target band it reports:

- brand agreement/conflict;
- model-code exact/near/conflict;
- numeric overlap/conflict;
- capacity/volume/storage;
- pack count and quantity/unit;
- size/dimensions;
- colour;
- gender/season;
- material/composition;
- year/generation;
- jewelry hallmark/karat/stone;
- main-product/accessory indicators;
- lexical similarity bins;
- title length and attribute density;
- attribute-key overlap;
- historical weak target margin.

Output: deterministic `STRATA.json` containing support and class balance for every usable `category × semantic_reason × difficulty` stratum. Electronics, Clothing, Footwear, Jewelry, Accessories and Furniture receive explicit tail reports; all 20 official categories remain in training and evaluation.

## 5. Human-grounded LLM audit

### 5.1 Calibration split

Within each outer-train human set, make a component/item-disjoint split into `model_train_human` and `llm_audit_human`. The audit subset is not used to optimize student weights during D5–D9 validation. After all model-selection gates are frozen, these rows may rejoin the authoritative human pool only for the final full-development production refit.

Sampling is stratified by official category, class and semantic reason, with extra support for rare critical conflicts.

### 5.2 Machine-validated teacher output

Every teacher invocation returns JSON with:

- `verdict`: `MATCH | NON_MATCH | UNCERTAIN`;
- `reason_code`: one of `SAME_MODEL`, `MODEL_CONFLICT`, `CAPACITY_CONFLICT`, `SIZE_CONFLICT`, `PACK_COUNT_CONFLICT`, `VARIANT_CONFLICT`, `ACCESSORY`, `DIFFERENT_GENERATION`, `BRAND_CONFLICT`, `SPARSE_EVIDENCE`, `OTHER`;
- extracted brand/model fields;
- critical attributes;
- conflicts;
- concise evidence grounded only in the two supplied item cards.

`UNCERTAIN` is never converted to a hard target. Self-reported model confidence is logged but never used for admission.

### 5.3 Strict two-teacher policy

LLM-generated labels are eligible for admission only when **two independently configured teacher lines are available**. Exact model IDs, revisions, provider/runtime, quantization and temperature are pinned before audit results.

Eligible acceptance requires teacher consensus plus deterministic structured-checker compatibility. Teacher disagreement or checker conflict goes to `active_review`.

There is no one-teacher exception. If a second teacher cannot be used reproducibly, v20 falls back to deterministic generated-pair supervision and existing audited weak signals; it does not admit new LLM-generated labels.

External APIs, if used for offline labelling, may not leak secrets into logs, public Git, private data manifests distributed with the final solution, or the submission ZIP. Final inference never requires them.

### 5.4 Statistical admission

For every category/reason stratum, compute precision on `llm_audit_human` and its two-sided 95% Wilson interval. Admission requires the lower confidence bound to exceed the predeclared floor:

- positive precision LCB >= 0.985;
- negative precision LCB >= 0.995;
- category aggregate precision LCB >= 0.970;
- critical conflict precision LCB >= 0.950.

Minimum support per stratum is fixed in the implementation plan before teacher audit labels are inspected. Insufficient-support strata are rejected rather than pooled post-result. Thresholds are never relaxed after seeing metrics.

## 6. Target-free candidate generation

### 6.1 Existing weak rows

The 11.2M historical weak pairs are primarily a candidate graph. Their `target` is a noisy teacher signal, not authoritative truth. v20 re-stratifies them by category/reason/difficulty instead of blindly ingesting all rows.

### 6.2 New real-item pairs

Generate candidate pairs from real item cards only, using deterministic blocks:

- same category;
- brand/model-code blocks;
- normalized title nearest neighbours;
- critical-attribute blocks;
- controlled near-model blocks;
- accessory/main-product blocks;
- same-product candidates under seller/title variation.

Negative candidates target model, capacity, size, generation, pack-count, gender/material and accessory conflicts. Positive candidates target same identity under sparse/reordered/noisy metadata.

Fictitious item records are not the primary source. Label-preserving text augmentation may be used only as a secondary regularizer and may not invent identity facts.

### 6.3 Diversity controls

Deterministic caps prevent domination by one anchor, item, duplicate cluster, category, class or semantic reason. Canonical duplicate pairs collapse to one row. Candidate budgets and per-stratum caps are fixed after D1 descriptive counts and before any v20 model-quality result.

## 7. Training architecture

### 7.1 Runtime backbone

- pinned `ai-forever/ruBert-base` revision;
- pair CrossEncoder;
- max length 256;
- one production checkpoint;
- no LLM, graph, TF-IDF ensemble or remote service at inference.

### 7.2 Multi-task rationale distillation

Training-only auxiliary heads share the RuBERT representation:

- main match head;
- model-conflict head;
- numeric-conflict head;
- variant-conflict head;
- accessory head;
- coarse reason-code head.

Auxiliary labels come only from deterministic extractors or admitted teacher rows. Missing auxiliary labels are masked. Production retains only the shared encoder and main match head by default, so inference remains one CrossEncoder forward pass.

### 7.3 Source-aware loss

`L = L_match + lambda_reason * L_reason + lambda_consistency * L_pair_symmetry`

Source weights:

- human = 1.0;
- admitted generated/LLM = empirical stratum reliability, capped below human;
- historical weak = v18 quality weight × stratum reliability;
- uncertain/disagreement = 0.

Exact lambdas, replay fractions and reliability transforms are frozen in the implementation plan before GPU quality results.

## 8. Curriculum

Phase A: broad admitted weak/generated supervision with category/class/reason balancing.

Phase B: mixed authoritative human + weak/generated replay.

Phase C: short lower-LR authoritative recovery with a fixed replay fraction.

Phase D: optional v19-style refresh only if v19 itself passes its preregistered human/weak/Brier/category gates.

This prevents the historical `weak -> human -> done` pattern while avoiding blind replay of unaudited weak labels.

## 9. Validation and external-anchor proxy

### Human axis

- immutable component-disjoint folds;
- official 20-category Macro AP;
- per-category AP and worst-category regression;
- folds 0 and 1 required before production;
- sealed gold unopened.

### Weak/test-like axis

Use the v17 item-disjoint weak holdout with both soft and hard targets. Report hard-target Macro AP, soft Brier, soft cross-entropy, per-category metrics and critical-stratum metrics. This axis is pseudo-supervision agreement, not a leaderboard estimate.

### External-anchor proxy

Before a proxy can select v20, evaluate frozen v7/v12/v13B/v14 prediction sets or checkpoints on identical proxy slices. A promotable aggregate diagnostic must reproduce the observed Public LB ordering directionally:

`v14 > v12 > v13B > v7`.

Candidate diagnostics: weak Macro AP, Brier, soft cross-entropy, hard-positive AP, hard-negative AP, critical-stratum AP and tail-category aggregate.

A diagnostic that reproduces the human fold-0 misranking remains diagnostic only. The four leaderboard scores are never used as row-level labels or to fit a supervised meta-model.

## 10. Experiment ladder

Every stage writes immutable metrics plus a decision manifest. Downstream stages stop when prerequisites fail.

- **D0 — ledger repair:** one machine-readable v1–v19 ledger with local metrics, external anchors, runtime, data recipe and rejection reason; repair stale `CURRENT.json` on the v20 branch.
- **D1 — full data census:** complete semantic audit, `STRATA.json`, candidate-volume estimates. Prefer hosted CPU/M1.
- **D2 — proxy calibration:** freeze which aggregate proxy diagnostics are promotable using v7/v12/v13B/v14 only.
- **D3 — two-teacher human audit:** run exact labelling pipeline on fold-safe human calibration; compute Wilson admission gates.
- **D4 — generated-pair corpus:** target-free real-item candidate generation, two-teacher labelling, statistical admission, private bronze/silver/gold manifests.
- **D5 — v20-A data-only:** proven RuBERT recipe + admitted new data, no auxiliary rationale heads.
- **D6 — v20-B rationale multi-task:** same data/exposure as D5 + auxiliary rationale heads.
- **D7 — v20-C mixed replay:** keeper of D5/D6 + source-aware mixed replay.
- **D8 — v20-D scaled data:** scale only retained mechanisms; do not add a new architecture simultaneously.
- **D9 — fold confirmation:** frozen candidate on human folds 0 and 1 plus weak/proxy diagnostics.
- **D10 — production:** all 285,210 development rows + retained admitted corpus, sealed gold untouched, one checkpoint, exact ZIP, organizer gate and SHA-256.

## 11. Promotion invariants

Exact numeric deltas are fixed in the implementation plan before D5 training. The policy must enforce:

1. no promotion on human fold0 alone;
2. no promotion when an accepted external-anchor proxy regresses materially;
3. no large tail-category regression for a tiny global gain;
4. weak-axis gain alone is insufficient when authoritative human/audited strata regress;
5. generated LLM data must beat an identical architecture/exposure baseline without those rows;
6. rationale heads must beat D5 at identical data/exposure;
7. production requires two-fold confirmation plus runtime preflight;
8. no gate is loosened after results.

## 12. Runtime and packaging

Final package requirements:

- one RuBERT-compatible production checkpoint by default;
- organizer-compatible `run.py` and metadata;
- no network dependency or API secret;
- generated training corpus excluded from ZIP;
- exact input order preserved;
- finite, non-degenerate predictions;
- ZIP integrity and SHA-256 recorded;
- organizer-shaped 1,000-row Check mandatory;
- public/private-size RTX runtime gate when feasible;
- setup/extraction included in authoritative wall timing under the established v12+ gate.

Training-only auxiliary heads are stripped unless they add no extra forward pass and a measured benefit justifies retaining them.

## 13. Compute and queueing

Public `Ansible_lab`: source, tests, specs/plans and non-sensitive deterministic manifests.

Private `gpu-dispatch`: RTX workflows and private outputs. All expensive lines use the existing single GPU concurrency lane with `queue: max`; v20 is queued after v18/v19 rather than displacing either.

Execution tiers:

- GitHub-hosted macOS/Ubuntu: syntax/tests, audits, deterministic candidate generation, small MPS micro-smokes when available;
- RTX 2060 SUPER: canonical RuBERT training and comparable metrics;
- private HF dataset: durable generated-data/checkpoint/submission storage when credentials work;
- persistent runner disk: fallback if Actions artifact quota blocks upload.

## 14. Stop/fallback rules

Close a line rather than patch around it when:

- the two-teacher audit cannot meet precision LCB floors for useful strata;
- generated data improves teacher agreement but harms authoritative human or accepted proxy axes;
- rationale multi-task does not beat identical data-only D5;
- a larger/different backbone cannot beat RuBERT under the runtime envelope;
- no candidate proxy reproduces known v7/v12/v13B/v14 external ordering;
- a stage would require opening sealed gold.

Infrastructure failures are recorded as infrastructure evidence, never negative model evidence.

If LLM audit fails, v20 falls back to deterministic real-item pair generation + audited historical weak data + retained v18/v19 mechanisms. It does not admit unaudited LLM labels.

## 15. Deliverables

- `ecup_matching/experiments/v20/LEDGER.json`
- `ecup_matching/experiments/v20/PLAN.md`
- `ecup_matching/experiments/v20/RESULTS.md`
- semantic audit and `STRATA.json`
- proxy calibration report
- LLM audit/admission report
- private generated-corpus manifests
- D5–D8 ablation evidence
- two-fold confirmation manifest
- production provenance manifest
- final `ecup-v20-audited-rationale-v7runtime-submission.zip`
- exact SHA-256 and organizer runtime report

## 16. Non-claims

v20 targets a substantial Public LB improvement but cannot guarantee `>0.5` before the platform evaluates the exact archive. Human AP, weak AP, Brier, proxy diagnostics and teacher precision are separate evidence axes, not substitute leaderboard scores.
