# E-CUP v20 — Audited Data-Centric Rationale Distillation

Date: 2026-08-19
Status: design approved in chat; implementation not started
Branch: `ecup-v20-data-centric`

## 1. Objective

Build a materially stronger product-matching submission by improving supervision density and quality over the weak-labelled item universe while retaining the proven single-checkpoint `ai-forever/ruBert-base` pair CrossEncoder runtime shape.

v20 is not another residual/graph/post-processing iteration. It is a data-centric training-system change with three goals:

1. audit and stratify existing weak supervision instead of treating confidence margin as the main notion of quality;
2. generate additional informative real-item pairs and admit only labels whose quality is empirically measured on authoritative human data;
3. distill both binary decisions and structured matching reasons into the same runtime-efficient RuBERT student.

The final submission must remain self-contained and offline. LLMs may be used only during data preparation/training. The current E-CUP FAQ explicitly permits LLM use for relabelling in the product-matching task; all libraries/software used in the final solution must satisfy the competition licensing requirements. No external API call, paid key, remote model, or second inference service is permitted in the submitted runtime.

## 2. Evidence from v1–v19 that constrains the design

### 2.1 Keep the RuBERT pair CrossEncoder family

The strongest useful architectural jump came from neural pair modelling, and the later strong `ai-forever/ruBert-base` CrossEncoder family substantially exceeded compact/alternative backbones locally. Granite, late-interaction/token-cross, graph and small residual families did not establish a better production parent.

Therefore v20 does not replace the runtime backbone unless a bounded ablation proves a clearly superior checkpoint under the same runtime envelope. The baseline v20 production shape is one RuBERT pair CrossEncoder, max length 256, one `.safetensors` checkpoint.

### 2.2 Do not optimize another tiny human fold-0 delta

Known Public LB anchors:

| Candidate | Human fold-0 | Public LB |
|---|---:|---:|
| v7 | 0.7023802626 | 0.3655833314 |
| v12 | 0.7059297810 | 0.3798116204 |
| v13B | 0.7086611386 | 0.3783781653 |
| v14 | 0.7065769714 | 0.3803270470 |

Human fold-0 misranks v12/v13B/v14. It remains a safety axis, not the primary promotion signal.

### 2.3 Weak supervision is under-used

The canonical weak corpus contains 11,187,780 pairs over 12,384,610 items. The historical v12+ path used `weak_final_rows=600000` and `weak_epochs=0.35`, about 210,000 examples seen, ~1.88% of the pool.

The human and weak supervision item populations are disjoint. Human labels cover 711,304 items (~5.31% of all 13,397,761 items); weak labels cover ~92.44%. Text/attribute density is similar across populations, so the local/Public gap is not explained by a trivially richer human population.

### 2.4 Weak-label quality was never actually audited against human truth

The historical LLM audit had zero shared items between weak and human pools, so positive/negative precision could not be measured. v20 fixes the methodology: the exact candidate LLM labelling pipeline is audited by re-labelling a held authoritative human calibration set, not by searching for overlap with the historical weak corpus.

### 2.5 Hard-negative-only curricula are not trusted

Naive static hard-negative weighting in v2 regressed. The model-mined hard-negative second stage in v3 was rejected. v20 therefore balances semantic reasons/difficulty strata and preserves broad replay instead of concentrating training on the most disagreeing negatives.

### 2.6 v17 measured a forgetting problem, but weak truth is uncertain

The completed v17 control measured weak-holdout AP `0.6973930799` after weak training and `0.6565798751` after human fine-tuning, while human fold-0 was `0.7017637364`. This establishes a large change on the weak axis, but weak targets are pseudo-labels and are not authoritative truth. v20 may reuse anti-forgetting mechanisms only when they pass both human and audited-stratum gates.

## 3. Data contracts

### 3.1 Immutable sources

- `items.parquet`: 13,397,761 item records.
- authoritative `matches.parquet`: 365,654 human-labelled pairs.
- historical `matches_llm.parquet`: 11,187,780 weak-labelled pairs with `id1,id2,target`.
- frozen five-fold component/item-disjoint human split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- sealed gold: 80,444 rows; never opened or scored for model selection.

No generated pair may cross a held human fold through either endpoint. For outer fold `k`, every generated/audited/training pair must exclude all item IDs belonging to fold `k` before candidate generation, LLM labelling, teacher scoring, mining, or training.

### 3.2 New v20 data layers

All generated data remain private and are materialized as reproducible manifests, never committed as raw competition data to public Git.

- `bronze/audit_candidates.parquet`: deterministic human calibration pairs and target-free weak/unlabelled candidate pairs.
- `silver/llm_labels.parquet`: raw structured LLM outputs plus model/provider/revision/prompt hashes.
- `silver/admitted_labels.parquet`: only labels passing empirical stratum admission gates.
- `gold/train_pairs_fold{k}.parquet`: fold-safe human + admitted existing weak + admitted generated pairs with source/reason weights.
- `gold/active_review.parquet`: disagreement/uncertain examples excluded from training but retained for diagnostics/future labelling.

Each row must carry provenance: `source`, `generator_version`, `stratum`, `reason_code`, `label_origin`, `teacher_model`, `teacher_revision`, `prompt_sha256`, `admission_policy_sha256`, and fold exclusion evidence.

## 4. Semantic data audit

The first v20 stage scans the complete existing weak corpus and relevant item fields. It does not train a model.

For each official category and target band, produce counts/distributions for:

- exact/normalized brand agreement and conflict;
- exact model-code match, near-model, model conflict;
- numeric token overlap/conflict;
- capacity/volume/storage conflict;
- pack count and quantity/unit conflict;
- size/dimension conflict;
- colour variant;
- gender/season;
- material/composition;
- year/generation;
- jewelry hallmark/karat/stone;
- main product vs accessory indicators;
- title lexical similarity bins;
- title length and attribute density;
- exact/partial attribute-key overlap;
- historical weak target margin.

The audit outputs a deterministic `STRATA.json` containing support and class balance for every usable `category × semantic_reason × difficulty` stratum.

Primary tail categories that must receive explicit reporting: Electronics, Clothing, Footwear, Jewelry, Accessories, Furniture. No category is excluded from training or metric reporting.

## 5. Human-grounded LLM audit

### 5.1 Calibration set construction

For each outer fold, split only the outer-train human pairs into:

- `model_train_human`;
- `llm_audit_human`.

The audit subset is component/item-disjoint from the model's human validation fold and is never used to optimize student weights in the ablation being evaluated. Sampling is stratified by official category, class and semantic reason, with extra support for rare/critical conflict strata.

### 5.2 Exact labelling schema

Every teacher invocation must return machine-validated JSON:

```json
{
  "verdict": "MATCH | NON_MATCH | UNCERTAIN",
  "reason_code": "SAME_MODEL | MODEL_CONFLICT | CAPACITY_CONFLICT | SIZE_CONFLICT | PACK_COUNT_CONFLICT | VARIANT_CONFLICT | ACCESSORY | DIFFERENT_GENERATION | BRAND_CONFLICT | SPARSE_EVIDENCE | OTHER",
  "same_product_type": true,
  "brand_left": "...",
  "brand_right": "...",
  "model_left": "...",
  "model_right": "...",
  "critical_attributes": {},
  "conflicts": [],
  "evidence": []
}
```

`UNCERTAIN` is never converted to a hard target. LLM self-reported confidence is logged if provided but is never an admission signal.

### 5.3 Teacher policy

Use at least two independently configured open-license teacher lines when practical. Exact model IDs, revisions, provider/runtime, quantization and temperature are pinned in a private manifest. Offline/local inference is preferred when feasible. If an external API is used for offline labelling, its secret must never be committed, logged, copied into the final ZIP, or required at inference.

Acceptance modes:

1. teacher consensus plus deterministic structured checker agreement; or
2. one teacher plus deterministic checker, only for a stratum whose empirical human precision gate is already passed.

Teacher disagreement and `UNCERTAIN` go to `active_review`, not training.

### 5.4 Statistical admission

Admission is by measured precision on `llm_audit_human`, separately for positive/negative and semantic strata. Use a two-sided Wilson interval; the lower 95% confidence bound must exceed the threshold.

Default floors:

- positive precision LCB >= 0.985;
- negative precision LCB >= 0.995;
- category aggregate precision LCB >= 0.970;
- critical conflict stratum precision LCB >= 0.950.

Minimum support is predeclared per stratum before labels are inspected. A stratum with insufficient support is rejected, not pooled opportunistically after results. No threshold is loosened post-result.

## 6. Target-free candidate generation

Candidate generation uses no unknown target and no held-fold label.

### 6.1 Existing weak-pair recovery

The 11.2M historical weak rows are primarily a candidate graph. Historical `target` is treated as one noisy teacher signal, not authoritative truth. Existing rows are re-stratified and sampled by semantic reason and category; v20 does not blindly ingest all 11.2M targets.

### 6.2 New real-item pairs

Generate additional pairs from real item cards using deterministic blocking:

- same category;
- brand/model-code blocks;
- normalized title token/character nearest-neighbour blocks;
- critical attribute blocks;
- controlled near-model substitutions;
- accessory/main-product lexical blocks;
- same-product candidate blocks across seller/title variation.

Negative candidate types include model, capacity, size, generation, pack-count, gender/material and accessory conflicts. Positive candidates focus on same identity under title/attribute sparsity, seller variation, reordered fields and partial metadata.

Do not synthesize fictitious item records as the primary source. v20 prioritizes new pairings of real competition items. Text augmentation may be used only as a secondary label-preserving regularizer and must not invent identity facts.

### 6.3 Diversity caps

No anchor, item, seller-like duplicate cluster, or semantic reason may dominate the generated corpus. Deterministic caps are enforced per category, item degree, reason and duplicate signature. Exact canonical duplicate pairs are collapsed.

## 7. Training architecture

### 7.1 Runtime backbone

Baseline and expected production architecture:

- `ai-forever/ruBert-base` pinned revision;
- pair CrossEncoder;
- max length 256;
- one production checkpoint;
- no LLM, graph, TF-IDF ensemble or remote service at inference.

### 7.2 Multi-task rationale distillation

During training only, attach auxiliary heads to the shared RuBERT representation:

- main `match` head;
- `model_conflict`;
- `numeric_conflict`;
- `variant_conflict`;
- `accessory`;
- coarse `reason_code` classification.

Auxiliary labels come only from deterministic extractors or admitted LLM-labelled examples. Missing auxiliary labels are masked, never guessed.

The production package retains only the shared encoder and main match head unless a measured runtime-free benefit from keeping an auxiliary scalar is explicitly demonstrated. Default production inference remains one match logit per pair.

### 7.3 Source-aware loss

Total training loss:

`L = L_match + lambda_reason * L_reason + lambda_consistency * L_pair_symmetry`

Source weights are explicit:

- authoritative human = 1.0;
- admitted generated/LLM = empirical stratum reliability weight capped below human;
- historical weak = quality-aware weight from v18 plus stratum/reliability factor;
- uncertain/disagreement = 0.

Exact lambda values are frozen in the implementation plan before real GPU quality results are inspected.

## 8. Curriculum

v20 does not use `weak -> human -> done`.

### Phase A — broad audited supervision

Train on admitted historical weak + admitted generated pairs with category/class/reason balancing.

### Phase B — mixed human replay

Train on authoritative human plus weak/generated replay in the same phase. Human examples remain the strongest source while replay prevents abrupt domain forgetting.

### Phase C — short authoritative recovery with replay

Short lower-LR recovery emphasizes human rows but retains a fixed weak/generated replay fraction. This fraction is frozen before quality results.

### Phase D — optional anti-forgetting refresh

Apply the v19 weak refresh only if v19 itself passes its preregistered human/weak/Brier/category gates. v20 must not inherit a rejected v19 mechanism merely because it exists.

## 9. Validation and leaderboard-proxy calibration

### 9.1 Human axis

- immutable component-disjoint outer folds;
- Macro AP over exactly 20 official categories;
- per-category AP and worst-category regression;
- folds 0 and 1 are mandatory confirmation before production;
- sealed gold unopened.

### 9.2 Weak/test-like axis

Use the v17 item-disjoint weak holdout but preserve both soft and hard targets. Report:

- Macro AP against thresholded weak targets;
- Brier score against soft targets;
- soft cross-entropy;
- per-category metrics;
- critical semantic-stratum AP/calibration.

This axis measures agreement with pseudo-supervision on unseen weak-population items and is never called a leaderboard estimate.

### 9.3 External-anchor proxy

Before using any new proxy for promotion, score frozen prediction sets/checkpoints for v7, v12, v13B and v14 on identical proxy slices. A candidate proxy must reproduce the observed Public LB ordering at least directionally:

`v14 > v12 > v13B > v7`.

Candidate proxy dimensions:

- weak hard Macro AP;
- weak soft Brier;
- weak soft cross-entropy;
- critical-stratum AP;
- hard-positive AP;
- hard-negative AP;
- tail-category aggregate.

Any proxy that reproduces the human fold-0 misranking is diagnostic only and cannot select v20.

Do not fit row-level targets to four leaderboard scores. External anchors are used only to reject misleading aggregate diagnostics, not as pseudo-labels or supervised meta-training data.

## 10. Experiment ladder

Each stage writes immutable metrics and a decision manifest. Later stages do not run when the prerequisite gate fails.

### D0 — experiment ledger repair

Create one v1–v19 machine-readable ledger with local metrics, Public LB anchors, runtime, data recipe, outcome and rejection reason. Fix stale `CURRENT.json` state on the v20 branch only. This prevents re-running rejected ideas.

### D1 — full data census

Run the complete weak/item semantic audit and generate `STRATA.json` plus candidate-volume estimates. CPU/M1/GitHub-hosted execution is preferred because no training is required.

### D2 — proxy calibration

Evaluate available frozen v7/v12/v13B/v14 checkpoints or immutable prediction vectors on the same candidate proxy axes. Freeze which proxy diagnostics are promotable before v20 model results.

### D3 — LLM human audit

Run exact teacher pipeline on the fold-safe human calibration set; compute Wilson gates per stratum. No generated label is admitted before this stage passes.

### D4 — generated-pair corpus

Generate target-free real-item candidate pairs, teacher-label them, apply statistical admission and publish private bronze/silver/gold manifests.

### D5 — v20-A data-only baseline

Existing proven RuBERT training recipe + admitted new data, no auxiliary rationale heads. This isolates data quality effect.

### D6 — v20-B rationale multi-task

Same data/exposure as D5 + auxiliary rationale heads. This isolates architectural benefit.

### D7 — v20-C source-aware mixed replay

Keeper of D5/D6 + mixed human/weak/generated replay curriculum. This isolates anti-forgetting/source-balance effect.

### D8 — v20-D scaled data

Scale only the mechanisms already individually retained. Increase admitted corpus exposure; do not introduce a new architecture simultaneously.

### D9 — fold confirmation

Run frozen candidate on human folds 0 and 1 plus weak/test-like diagnostics. Both folds must pass the preregistered gate.

### D10 — production refit and package

Full 285,210 development rows + retained admitted corpus, sealed gold untouched. Save one checkpoint, build exact organizer ZIP, run exact organizer-shaped Check and record SHA-256.

## 11. Promotion policy

Exact numeric deltas for D5–D9 are fixed in the implementation plan before training. The policy must satisfy these invariants:

1. no candidate may be promoted on human fold0 alone;
2. no candidate may be promoted when an accepted external-anchor proxy regresses materially;
3. no candidate may trade a large tail-category regression for a tiny global gain;
4. weak-axis gain alone is insufficient when audited human strata regress;
5. LLM/generated data must demonstrate an admitted-data ablation gain over the same architecture/data exposure without those rows;
6. multi-task rationale heads must demonstrate an ablation gain over D5 at the same data;
7. production is allowed only after two-fold confirmation and runtime preflight.

No gate is loosened after results.

## 12. Runtime and packaging contract

Final package requirements:

- one RuBERT-compatible production checkpoint by default;
- `run.py` and metadata contract compatible with organizer image;
- no network dependency;
- no LLM/API/provider secret;
- no training-only raw generated corpus in ZIP;
- exact input pair order preserved;
- finite, non-degenerate predictions;
- ZIP integrity and SHA-256 recorded;
- organizer-shaped 1,000-row Check mandatory;
- full public/private-size runtime gate mandatory when feasible on the RTX 2060S;
- archive setup/extraction included in authoritative wall timing where the established v12+ gate includes it.

If multi-task training changes only training-time heads, strip auxiliary-only weights/modules from production or prove they add no extra inference forward pass.

## 13. Compute/orchestration

Public `Ansible_lab` contains source, tests, design/plan, deterministic manifests without raw sensitive competition data.

Private `gpu-dispatch` contains self-hosted RTX workflows and private outputs. Use a single concurrency group with `queue: max` so v18/v19/v20 are not displaced.

Execution tiers:

- GitHub-hosted macOS/Ubuntu: syntax, unit tests, deterministic audits, small corpus/sample generation, MPS micro-smokes when available;
- RTX 2060 SUPER: canonical RuBERT training and comparable model metrics;
- private HF dataset: durable generated-data manifests/checkpoints/submission artifacts when credentials are available;
- persistent runner disk: fallback when GitHub artifact quota blocks upload.

## 14. Failure and stop rules

Close rather than patch around a line when:

- LLM audit cannot achieve the precision LCB floor for useful strata;
- generated data improves only the teacher-agreement metric but harms authoritative human or accepted proxy axes;
- rationale multi-task does not beat the identical data-only baseline;
- a larger/different backbone cannot fit runtime/VRAM or cannot beat RuBERT under equal data;
- proxy diagnostics cannot reproduce known v7/v12/v13B/v14 external ordering;
- a stage requires opening sealed gold;
- an infrastructure failure occurs: record it as infrastructure evidence, never as negative model evidence.

If LLM audit fails broadly, v20 falls back to deterministic generated-pair supervision plus v18/v19 retained training mechanisms; it does not admit unaudited LLM labels.

## 15. Expected deliverables

- `ecup_matching/experiments/v20/LEDGER.json`
- `ecup_matching/experiments/v20/PLAN.md`
- `ecup_matching/experiments/v20/RESULTS.md`
- full semantic audit manifest
- proxy calibration report
- LLM audit/admission report
- private generated corpus manifests
- ablation results D5–D8
- two-fold confirmation manifest
- production provenance manifest
- final `ecup-v20-audited-rationale-v7runtime-submission.zip`
- exact archive SHA-256 and organizer runtime report

## 16. Non-claims

v20 targets a substantial Public LB improvement but cannot guarantee `>0.5` before the competition platform evaluates the exact archive. Local human AP, weak AP, Brier, proxy diagnostics and teacher precision are evidence axes, not substitute leaderboard scores.
