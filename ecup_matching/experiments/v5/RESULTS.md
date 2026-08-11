# E-CUP Matching — Iteration v5 Results

Date: 2026-08-11
Status: **in progress — sealed gold has not been scored**

## External leaderboard evidence motivating v5

Observed hidden Macro AP supplied by the participant:

| Submission | Hidden Macro AP |
|---|---:|
| v1 | 0.23458522924335687 |
| v2 | **0.2583231811423486** |
| v3 non-canonical | 0.2583231811423486 |
| v3 canonical | 0.24810151893254498 |
| v4 canonical | 0.2531285194869718 |

The hidden-best production anchor is v2. Old local v3/v4 gains are historical offline evidence only.

## v5a — validation audit

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
- gold metric opened: **false**.

Baseline used for the audit is a human-only structured model with the existing v2 feature family but without label-derived category attribute importance. Its purpose is to measure split stability, not to replace exact v2b.

Five development-fold Macro AP values:

1. `0.535137201283413`
2. `0.5293875046954674`
3. `0.5375146978687112`
4. `0.531512424671052`
5. `0.5337619678140229`

Aggregate:

- development OOF Macro AP: **`0.5315527708634168`**;
- fold mean: **`0.5334627592665332`**;
- fold median: `0.5337619678140229`;
- fold standard deviation: **`0.0028186164376135885`**;
- worst fold: **`0.5293875046954674`**.

The low fold spread is materially more useful for model development than the old single repeatedly reused holdout. The user-requested `0.60` stretch target requires about `+0.06845` absolute Macro AP over this OOF baseline, so it cannot responsibly be pursued through calibration-only changes.

Development OOF AP by category:

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

Workflow run: `31481563972`
Job: `93747448310`
Source/workflow commit: `e885961388d156aeb7a28f111c7cb726c7f383bc`
Private prefix: `experiments/v5/category/e885961388d1`
Safe metrics artifact ID: `9097526746`

Instead of one global HGB sharing tree capacity across all 20 product regimes, this ablation trains one compact HGB per category using the same label-safe structured feature family. No category alpha or post-hoc calibration is fit.

Result:

- audit base OOF: `0.5315527708634168`;
- **category-specialist OOF: `0.5476778075943867`**;
- **delta: `+0.016125036730969877`**.

Held-fold Macro AP:

1. `0.554183074700053`
2. `0.54539395716035`
3. `0.5467701244203692`
4. `0.54898246234179`
5. `0.5464930969078342`

The improvement is stable across all five held folds. This is the first retained v5 development improvement and the current development-best score. Gold remains sealed.

Per-category OOF AP:

| Category | AP |
|---|---:|
| Автотовары | 0.5198201157 |
| Аптека | 0.5264467916 |
| Бытовая техника | 0.7015272270 |
| Бытовая химия | 0.7038625946 |
| Галантерея и аксессуары | 0.3968907770 |
| Детские товары | 0.7858681147 |
| Дом и сад | 0.5657858318 |
| Канцелярские товары | 0.5641442356 |
| Красота и гигиена | 0.6032217267 |
| Мебель | 0.4469702270 |
| Музыкальные инструменты | 0.6867889472 |
| Обувь | 0.3125024396 |
| Одежда | 0.3407365496 |
| Продукты питания | 0.5853524837 |
| Спорт и отдых | 0.4874632584 |
| Строительство и ремонт | 0.5488082642 |
| Товары для животных | 0.6279183237 |
| Хобби и творчество | 0.8059908035 |
| Электроника | 0.4435916619 |
| Ювелирные изделия | 0.4038659838 |

Decision: **KEEP**. Future semantic/contrastive candidates are evaluated on top of these OOF category-specialist scores, not the weaker global audit base.

## v5b — direct attribute log-likelihood evidence — REJECT

Workflow run: `31482167758`
Job: `93749392995`
Source/workflow commit: `f19e58475dd877dc1e1c27b4671b26b8e981d902`
Private prefix: `experiments/v5/attribute/f19e58475dd8`
Safe metrics artifact ID: `9097744495`

This ablation learned category/key-specific equal-vs-different log likelihood evidence on each outer-train partition and added that evidence to the base logit with no held-fold calibration.

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

Decision: **REJECT**. The direct evidence is too aggressive for ranking and is not rescued by tuning a post-hoc scale on the development folds. A safer follow-up may expose fold-trained attribute importance as ordinary features to category-specialist trees instead of directly changing the score.

## Implemented v5 foundations

- balanced component-disjoint split manifest with deterministic hash;
- hard gold-evaluation eligibility checks and candidate freeze hashes;
- paired component bootstrap;
- v2-anchored residual correction primitive;
- identity-first item serializer preserving name/brand/model/numeric/attribute sections;
- compact symmetric dense-embedding pair features;
- train-only sparse TF-IDF item representation with unseen-item transform support;
- strict OOF semantic second-level stack;
- leakage-safe OOF hard-negative curriculum for supervised item-space training.

## Current state

Current development best: **`0.5476778075943867`** from category specialists.

Still running / under evaluation:

- pretrained multilingual item bi-encoder transfer;
- five-fold supervised contrastive item-encoder sprint;
- fold-specific learned attribute importance used inside specialist trees.

The sealed gold score remains unknown. No v5 submission candidate is retained yet.
