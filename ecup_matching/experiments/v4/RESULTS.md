# E-CUP Matching — Iteration v4 Results

Date: 2026-08-11
Status: **completed / retained**

## Retained result

v4 retains the **same immutable v3 structured + neural models** and replaces the single global neural blend weight with a regularized per-category routing vector selected by leakage-aware cross-fitting.

This is intentionally distinguished from the stronger `ai-forever/ruBert-base` training ladder that was implemented during v4 but did not produce the retained artifact.

Fixed evaluation basis remains the same human item-disjoint validation:

- rows: `73,131`;
- train/validation item overlap: `0`;
- categories: `20`;
- connected validation item-components used for blend cross-fit: `53,131`.

Retained v3 baseline Macro AP: `0.5254642645846543`.

### Honest model-selection score

Selection protocol: **5-fold `GroupKFold` by connected components of all validation candidate edges**. A held-out component never contributes labels to the category alpha used to score that component.

Regularization prior candidates: `250, 500, 1000, 2000, 4000, 8000`.

Selected prior: **`4000`**.

- cross-fitted global-blend Macro AP: `0.526005894031544`;
- **cross-fitted regularized category-blend Macro AP: `0.5276431099433088`**;
- honest absolute delta vs retained v3: **`+0.0021788453586544243`**.

The cross-fitted value is the headline quality number for v4 because it measures the routing rule out of fold. It is deliberately reported instead of the larger full-fit number as the unbiased retention evidence.

Cross-fit selection run: `31473553650`.

### Final deployable coefficients

After the regularizer strength was selected exclusively from OOF predictions, the deployable category alphas were fit on all `73,131` labelled validation rows. That final coefficient fit scores:

- full-fit Macro AP: `0.5284493942551521`;
- full-fit delta vs v3: `+0.002985129670497799`;
- full-data global neural alpha anchor: `0.4`.

This full-fit score describes the packaged coefficients; it is **not** substituted for the cross-fitted headline generalization estimate.

Final neural alphas by category:

| Category | Neural alpha |
|---|---:|
| Автотовары | 0.3117573317125057 |
| Аптека | 0.4720171562256304 |
| Бытовая техника | 0.3887211855104281 |
| Бытовая химия | 0.33314629608093965 |
| Галантерея и аксессуары | 0.4531609195402299 |
| Детские товары | 0.4979781088476741 |
| Дом и сад | 0.42858192789726135 |
| Канцелярские товары | 0.430797848828643 |
| Красота и гигиена | 0.4291634835752483 |
| Мебель | 0.3363873542050338 |
| Музыкальные инструменты | 0.17052341597796145 |
| Обувь | 0.4 |
| Одежда | 0.4 |
| Продукты питания | 0.4121630295250321 |
| Спорт и отдых | 0.42958673932788377 |
| Строительство и ремонт | 0.4 |
| Товары для животных | 0.3769323996768112 |
| Хобби и творчество | 0.4596521608191538 |
| Электроника | 0.45747847347719783 |
| Ювелирные изделия | 0.41511369253583785 |

Full-fit per-category AP:

| Category | AP |
|---|---:|
| Автотовары | 0.520386970905967 |
| Аптека | 0.5564134722059562 |
| Бытовая техника | 0.6672385698089135 |
| Бытовая химия | 0.6834399696407997 |
| Галантерея и аксессуары | 0.3951237826216045 |
| Детские товары | 0.771530605360941 |
| Дом и сад | 0.541601821160062 |
| Канцелярские товары | 0.549669889995236 |
| Красота и гигиена | 0.6052257967302181 |
| Мебель | 0.38343362188565294 |
| Музыкальные инструменты | 0.6282624855903924 |
| Обувь | 0.3059786401343868 |
| Одежда | 0.29192679041443326 |
| Продукты питания | 0.5665688003991547 |
| Спорт и отдых | 0.48280015084993577 |
| Строительство и ремонт | 0.5206552610357673 |
| Товары для животных | 0.6064625162841497 |
| Хобби и творчество | 0.8145133207738111 |
| Электроника | 0.33069066257923246 |
| Ювелирные изделия | 0.34706475672642795 |

## Canonical package

Freeze workflow: `31474888023`, job `93726203398`.

Source commit used by the freeze workflow: `6e80453053ee1738e8c2e4e351132fb13ba58f0e`.

Immutable inputs:

- v3 ZIP SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`;
- frozen v3 validation-prediction SHA-256: `4112aa2556cb683ffca27cd9bd16c00a7149bb7e3279d1f2a6abb2b20438d643`.

Final v4 package:

- ZIP bytes: **`109,185,879`**;
- SHA-256: **`b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`**;
- canonical private prefix: `submissions/v4/canonical/b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`;
- convenience alias: `submissions/v4/ecup-v4-submission.zip`;
- metrics alias: `submissions/v4/v4-package-metrics.json`.

The freeze workflow verified the immutable source hashes, reproduced the exact expected full-fit score before packaging, and refused to freeze before runtime verification.

## Organizer runtime verification

Exact organizer image: `odsai/ecup26-matching-baseline:1.0`.

Offline smoke conditions:

- `--network none`;
- read-only container root;
- submission/runtime inputs mounted read-only;
- 1,000 input pairs;
- **1,000 / 1,000 neural pairs actually routed through the neural model**;
- output rows/order/schema/range/finite checks passed;
- unique prediction values: `1,000`;
- smoke runtime log: structured `1,000/1,000`, neural `1,000/1,000`, valid `submit.csv` written;
- canonical private upload/presence verification: PASS.

The hosted smoke had no NVIDIA driver and therefore exercised the organizer image on CPU. This is a correctness/offline-compatibility gate; the v3 runtime selects CUDA automatically when it is available in the organizer environment.

## Stronger-encoder v4 branch — not retained

The original v4 plan also implemented a pinned `ai-forever/ruBert-base` staged ladder:

- v4a: complete authoritative human curriculum;
- v4b: confidence-filtered weak-label continuation;
- v4c: model-mined hard negatives with 50% ordinary replay.

A first private RTX production attempt (`31470932265`) terminated with exit `137` during the host-memory-heavy preparation path before any quality metric existed. It was **not** recorded as a negative ML result.

The production code was subsequently hardened:

- >11M weak rows are streamed with PyArrow instead of fully materialized in pandas;
- structured/curriculum preparation occurs before loading the 178M-parameter BERT;
- serialized weak-pair direction is regression-tested;
- private v4 Docker RAM is fail-contained at 10 GiB with no extra swap.

The home WSL runner went offline after the original host shutdown. A separate GitHub Apple-Silicon/MPS strong-encoder diagnostic remained non-canonical while this retained v4 was frozen. No `ruBert-base` metric is attributed to the retained v4 artifact.

## Decision

**Retain v4.**

Why:

1. the category-routing regularizer was chosen by component-disjoint OOF evaluation;
2. the honest cross-fitted Macro AP `0.5276431099433088` strictly exceeds retained v3 `0.5254642645846543`;
3. the final coefficients reproduce their expected score from immutable frozen predictions;
4. the exact organizer-image offline runtime gate passed with real neural routing;
5. the canonical package is immutable and privately frozen by SHA-256.

v3 remains an immutable fallback, but v4 is now the current best submission candidate.