# E-CUP Matching — Iteration v5 Results

Date: 2026-08-11
Status: **in progress — sealed gold has not been scored; no v5 submission retained yet**

## External leaderboard evidence motivating v5

Observed hidden Macro AP supplied by the participant:

| Submission | Hidden Macro AP |
|---|---:|
| v1 | 0.23458522924335687 |
| v2 | **0.2583231811423486** |
| v3 non-canonical | 0.2583231811423486 |
| v3 canonical | 0.24810151893254498 |
| v4 canonical | 0.2531285194869718 |

**Interpretation:** v2 is the immutable production/leaderboard fallback during v5 development. Old v3/v4 local gains are historical offline evidence only; they must not be treated as the primary proxy for hidden transfer.

## v5a — sealed validation audit

Workflow run: `31479778679`
Job: `93741817398`
Source commit: `93e4396330997c41bfb309f449f1dcb79a5e4db6`
Private prefix: `experiments/v5/validation/93e439633099`
Safe metrics artifact ID: `9096981220`

Split:

- authoritative human rows: `365,654`;
- connected item components: `345,654`;
- development rows: `285,210`;
- sealed gold rows: **`80,444`**;
- development folds: `5`;
- cross-split item overlap: **`0`**;
- split SHA-256: **`aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`**;
- gold metric opened: **false**;
- gold rows scored: **0**.

Baseline used for the audit is a human-only structured model with the existing v2 feature family but without label-derived category attribute importance. Its purpose is to measure split stability, not to reproduce exact v2b.

Five development-fold Macro AP values:

1. `0.535137201283413`
2. `0.5293875046954674`
3. `0.5375146978687112`
4. `0.531512424671052`
5. `0.5337619678140229`

Aggregate:

- development OOF Macro AP: **`0.5315527708634168`**;
- fold mean: `0.5334627592665332`;
- fold median: `0.5337619678140229`;
- fold standard deviation: **`0.0028186164376135885`**;
- worst fold: **`0.5293875046954674`**.

The low spread makes this materially safer for model development than repeatedly optimizing one old holdout. The requested `0.60` stretch target must be reached on this immutable development protocol, not by recalibrating/re-splitting until a desired number appears.

Baseline OOF AP by category:

| Category | AP |
|---|---:|
| Автотовары | 0.5036164299 |
| Аптека | 0.5352936984 |
| Бытовая техника | 0.6872407144 |
| Бытовая химия | 0.6910049823 |
| Галантерея и аксессуары | 0.3771211063 |
| Детские товары | 0.7660074930 |
| Дом и сад | 0.5424579108 |
| Канцелярские товары | 0.5577463393 |
| Красота и гигиена | 0.5809546318 |
| Мебель | 0.4103383638 |
| Музыкальные инструменты | 0.6802530439 |
| Обувь | 0.2734579960 |
| Одежда | 0.2870273768 |
| Продукты питания | 0.5600314238 |
| Спорт и отдых | 0.4664615540 |
| Строительство и ремонт | 0.5348013880 |
| Товары для животных | 0.6205790307 |
| Хобби и творчество | 0.7902910115 |
| Электроника | 0.4117454803 |
| Ювелирные изделия | 0.3546254423 |

## v5b — category-specialist structured models — KEEP

Two closely related category-specialist executions exist. The canonical downstream OOF file used by later stack/weak workflows reports:

- audit base OOF: `0.5315527708634168`;
- **category-specialist OOF: `0.5476780661335778`**;
- **delta: `+0.016125295270161044`**;
- private prefix: `experiments/v5/category/e885961388d1`.

Instead of one global HGB sharing tree capacity across all 20 product regimes, this trains one compact HGB per category using the same label-safe feature family. No category alpha/post-hoc calibration is needed.

Canonical downstream held-fold Macro AP:

1. `0.5504494811533941`
2. `0.5478362369858445`
3. `0.554178032571296`
4. `0.5453862553120211`
5. `0.5500737968478944`

All five folds improve relative to the audit baseline. **KEEP** as the v5 structured base; future candidates are judged for incremental information above this score.

An earlier safe-metrics snapshot from run `31481563972` recorded `0.5476778075943867`; the tiny numerical difference comes from a closely related execution and must not be confused with a conceptual model change. Downstream v5 workflows consistently use the `0.5476780661335778` canonical category OOF file.

## v5b — direct attribute log-likelihood evidence — REJECT

Workflow run: `31482167758`
Job: `93749392995`
Source/workflow commit: `f19e58475dd877dc1e1c27b4671b26b8e981d902`
Private prefix: `experiments/v5/attribute/f19e58475dd8`
Safe metrics artifact ID: `9097744495`

This learned category/key-specific equal-vs-different log-likelihood evidence on each outer-train partition and added it directly to the base logit.

Result:

- base OOF: `0.5315527708634168`;
- attribute-evidence OOF: `0.523218903672764`;
- delta: **`-0.008333867190652766`**.

Every held fold regressed:

- fold 0: `-0.004407802913674264`;
- fold 1: `-0.006267433707208281`;
- fold 2: `-0.010836069631330214`;
- fold 3: `-0.00942910890384685`;
- fold 4: `-0.009296400437025731`.

**Decision: REJECT.** Do not “rescue” this by fitting a scale on the same development folds. Attribute evidence may enter as ordinary estimator features instead of an unconditional direct score shift.

## v5c — pretrained multilingual item bi-encoder — insufficient standalone

Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` item-space diagnostic.

Result:

- audit base OOF: `0.5315527708634168`;
- stack OOF: **`0.5318080650341337`**;
- delta: `+0.0002552941707169021`;
- raw semantic cosine Macro AP: about `0.31201359969464276`;
- all held folds were slightly positive, but the gain is too small to retain standalone;
- gold items encoded: `0`; gold metric opened: false.

**Decision:** pretrained item-space transfer is diagnostic only. Supervised/weak task adaptation must prove OOF value above the retained category base.

## v5b2 — fold-weighted category specialists — diagnostic only

Workflow run: `31483353777`
Source: `9df24f7ee1335046d00751d4c2b0cbd78b00e162`
Private aggregate: `experiments/v5/weighted-specialists/9df24f7ee133/aggregate`
Metrics artifact ID: `9098324849`
Artifact ZIP digest: `ec9614c9906cf4b9c39c613a96c73294d885158d0bd3ec358b911600ca13cea9`

Result:

- category base: `0.5476780661335778`;
- weighted-specialist OOF: **`0.5498696731704964`**;
- delta: `+0.0021916070369185636`.

Held-fold deltas vs category base:

- fold 0: `+0.0021354161959221685`;
- fold 1: `+0.004859795667646272`;
- fold 2: `-0.00031919882576014746`;
- fold 3: `-0.0009191637590409973`;
- fold 4: `+0.004585239432923327`.

**Decision:** do not retain standalone because folds 2 and 3 regress. Keep the OOF predictions as a potential stack input only.

## v5e — leakage-safe weak category specialists — KEEP

Workflow run: **`31484641329`**
Source: **`319993a469cfa37770d66cfaf1b2203515dc9841`**
Private aggregate: `experiments/v5/weak-specialists/319993a469cf/aggregate`
Metrics artifact ID: **`9099098118`**
Artifact ZIP digest: `3f8b2db519f4f6925a8cc6f7bffb7cde7869d09c50a12e5be957c0894089ee93`

Fold curriculum invariants:

- weak input rows: `11,187,780`;
- confidence-filtered presample cap: `250,000`;
- final weak rows per fold: `150,000`;
- held-fold items forbidden from weak: **true**;
- sealed-gold items forbidden: **true**;
- gold rows used/scored: `0`.

Result:

- category base OOF: `0.5476780661335778`;
- **weak-specialist OOF: `0.5514237338676234`**;
- **delta: `+0.00374566773404561`**.

Held-fold results:

| Fold | Base AP | Weak AP | Delta |
|---:|---:|---:|---:|
| 0 | 0.5504494811533941 | 0.5546298909754048 | +0.004180409822010733 |
| 1 | 0.5478362369858445 | 0.5520443319510406 | +0.004208094965196141 |
| 2 | 0.554178032571296 | 0.5570904113247204 | +0.002912378753424316 |
| 3 | 0.5453862553120211 | 0.5470536921958663 | +0.0016674368838451858 |
| 4 | 0.5500737968478944 | 0.5551003946611213 | +0.005026597813226896 |

All five folds improve. **KEEP**. This is strong evidence that weak supervision still transfers when item leakage is explicitly blocked.

## v5 combo — category + weighted + pretrained, strict meta-OOF — CURRENT DEVELOPMENT BEST / KEEP

Workflow run: **`31485240666`**
Source: **`7a1c1764a2bdda8f007b9bfea7d088911623e7f0`**
Private prefix: **`experiments/v5/combo/7a1c1764a2bd`**
Metrics artifact ID: **`9098856613`**
Artifact ZIP digest: `b020be418c698384c45f0b3bf9c3a5acfb1e0d66eea79ad5f2caf4930dc86ede`

Inputs are already-OOF category-specialist, fold-weighted and pretrained semantic signals. The meta-estimator is itself cross-fitted: each held fold is scored by a stack trained only on the other development folds. It never trains on a row's target together with that row's own in-sample base prediction.

Result:

- category base OOF: `0.5476780661335778`;
- **combo OOF: `0.559512531439709`**;
- **delta vs category base: `+0.011834465306131192`**;
- delta vs audit baseline: `+0.0279597605762922`;
- gold metric opened: **false**;
- gold rows scored: **0**.

Held-fold results:

| Fold | Base AP | Combo AP | Delta |
|---:|---:|---:|---:|
| 0 | 0.5504494811533941 | 0.562580065789817 | +0.012130584636422914 |
| 1 | 0.5478362369858445 | 0.5605549739646596 | +0.012718736978815093 |
| 2 | 0.554178032571296 | 0.5667063354890245 | +0.01252830291772844 |
| 3 | 0.5453862553120211 | 0.5579194708823281 | +0.012533215570306955 |
| 4 | 0.5500737968478944 | 0.5631351691480011 | +0.013061372300106733 |

Every fold improves by roughly `+0.012`, making this stronger evidence than the weighted branch alone. **KEEP as the current v5 development benchmark.** It is still not a submission candidate until the agreed freeze/gold/runtime gates are passed.

Per-category combo OOF AP:

| Category | AP |
|---|---:|
| Автотовары | 0.5369454234 |
| Аптека | 0.5844749636 |
| Бытовая техника | 0.6965517470 |
| Бытовая химия | 0.7029196269 |
| Галантерея и аксессуары | 0.4130128019 |
| Детские товары | 0.7768827927 |
| Дом и сад | 0.5653335919 |
| Канцелярские товары | 0.5674996108 |
| Красота и гигиена | 0.5947046036 |
| Мебель | 0.4651673941 |
| Музыкальные инструменты | 0.6916668132 |
| Обувь | 0.3271995218 |
| Одежда | 0.3446915003 |
| Продукты питания | 0.5882027763 |
| Спорт и отдых | 0.4870704992 |
| Строительство и ремонт | 0.5573098961 |
| Товары для животных | 0.6353367621 |
| Хобби и творчество | 0.8059318564 |
| Электроника | 0.4555696478 |
| Ювелирные изделия | 0.3937787998 |

## Active experiments — metrics must not be invented

At this memory snapshot the following workflows have not produced a final aggregate metric:

- supervised contrastive item encoder outer-CV: run `31483288887` — **in progress**;
- `ai-forever/ruBert-base` pair teacher outer-CV: run `31485127564` — **in progress**;
- strict train-only sparse TF-IDF specialists: run `31485396599` — **in progress**;
- explicit per-key attribute specialists: run `31485990777` — **in progress**;
- field-aware weak ranking teacher: run `31486298300` — **queued** at this snapshot.

Do not convert any of these statuses into a model-quality conclusion until all held folds and the aggregate metric exist.

## Implemented v5 foundations

- balanced component-disjoint split manifest with deterministic hash;
- hard gold-evaluation eligibility checks and candidate freeze hashes;
- paired component bootstrap;
- v2-anchored residual correction primitive;
- identity-first item serializer preserving name/brand/model/numeric/attribute sections;
- compact symmetric dense-embedding pair features;
- train-only sparse TF-IDF item representation with unseen-item transform support;
- strict OOF semantic second-level stack;
- leakage-safe OOF hard-negative curriculum for supervised item-space training;
- sample-weight support in category specialists for weak/human source weighting;
- explicit-key and field-aware/ranking teacher experimental paths.

## Debugging lessons / “do not repeat”

### 1. MPS contrastive OOM is an infrastructure failure, not a rejected model

Physical batch `96` exhausted the available MPS memory (roughly `7.93 GiB / 7.93 GiB`) before meaningful training evidence. The corrected trainer keeps **effective batch 96** with physical microbatch `24` and gradient accumulation `4`. Do not mark contrastive learning REJECT because of the original OOM.

### 2. Weak-table preprocessing must remain bounded-memory

The v4/v5 path must not materialize all `>11M` weak rows in pandas before model loading. Use deterministic PyArrow streaming/two-pass selection; perform CPU-heavy preparation before loading a large Transformer.

### 3. TF-IDF unit tests must test representation contracts, not desired ranking outcomes

An early test incorrectly expected raw TF-IDF to infer that an unknown SKU/color conflict must rank in a specific order. Train-only vocab naturally drops OOV tokens. Correct tests cover unseen `transform`, symmetry and finite/bounded outputs; **outer-fold AP determines usefulness**.

### 4. Attribute likelihood should not directly shift logits

The idea was plausible but empirically harmful on every fold. Do not reintroduce it with a tuned coefficient on the same folds. If attribute-key specificity is useful, feed it to a model as features.

### 5. Read the exact failing test before assigning blame

A serializer commit appeared red because the next deliberately RED embedding test had already been added; the serializer itself later passed unchanged. Overlapping TDD cycles can make the most recent implementation look guilty when another intentionally missing module is the actual failure.

### 6. Memora can miss an important result when docs land during RED TDD

Memora runs `31481012401` and `31482891498` failed **before ingest** because their memory-triggering commits occurred while the full workspace was intentionally RED. In `31482891498`, collection failed on `ModuleNotFoundError: ecup_matching.ml.v5_weighted_specialists`. Later GREEN code did not retroactively run the skipped memory checkpoint.

Do **not** weaken Memora's full-test gate. Instead ensure retained docs are checkpointed from a GREEN state; if a memory update happened during RED, make a later memory-triggering commit/manual dispatch once GREEN.

## Current state

- **Production/hidden fallback:** v2, hidden Macro AP `0.2583231811423486`.
- **Development baseline:** `0.5315527708634168` OOF.
- **Retained structured base:** category specialists `0.5476780661335778` OOF.
- **Retained weak branch:** `0.5514237338676234` OOF, all folds positive.
- **Current development best:** **combo `0.559512531439709` OOF**, all folds positive.
- **Stretch target:** `0.60` honest development OOF on the same immutable split.
- **Sealed gold:** still unopened, `80,444` rows, zero scored.
- **v5 submission:** not retained yet.

Next work must continue from this state; never revert to the old 73,131-row holdout as the primary v5 selection gate and never open sealed gold merely to guide the next experiment.
