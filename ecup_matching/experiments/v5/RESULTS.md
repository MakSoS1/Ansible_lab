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

## Implemented v5 foundations

- balanced component-disjoint split manifest with deterministic hash;
- hard gold-evaluation eligibility checks and candidate freeze hashes;
- paired component bootstrap;
- v2-anchored residual correction primitive;
- identity-first item serializer preserving name/brand/model/numeric/attribute sections;
- compact symmetric dense-embedding pair features;
- train-only sparse TF-IDF item representation with unseen-item transform support.

No v5 quality candidate has yet been retained and the sealed gold score remains unknown.
