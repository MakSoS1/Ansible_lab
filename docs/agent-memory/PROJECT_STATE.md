# E-CUP Matching — Canonical Project State

Updated: 2026-08-11

## Objective

Score strongly in the ODS E-CUP 2026 Ozon pairwise product matching task while keeping model selection honest for unseen products, the final submission reproducible/offline-compatible, and all competition artifacts private.

## Fixed task/evaluation facts

- 20 product categories.
- Official metric: unweighted macro mean of `sklearn.metrics.average_precision_score` across categories.
- Human labels: `365,654` pairs; soft LLM labels: `>11M` pairs.
- Hidden test contains new/unseen products; item leakage is unacceptable.
- Organizer image: `odsai/ecup26-matching-baseline:1.0`.
- Public Git contains code/docs only. Raw data, learned models, OOF predictions, Memora DBs and submission ZIPs remain private.
- Private artifact/data repo: `Maksim123321/e-cup-2026-matching-private`.

## Never merge these two meanings of “best”

### Production / hidden-leaderboard anchor

Best observed hidden score among submitted v1-v4 candidates is **v2**:

| Submission | Hidden Macro AP |
|---|---:|
| v1 | 0.23458522924335687 |
| v2 | **0.2583231811423486** |
| v3 non-canonical | 0.2583231811423486 |
| v3 canonical | 0.24810151893254498 |
| v4 canonical | 0.2531285194869718 |

Therefore v2 is the immutable production/leaderboard fallback while v5 is being developed. Old v3/v4 local gains remain valid historical measurements but did not transfer monotonically to hidden evaluation.

### Current development best

v5 is **in progress**. Current strongest honest development result is the supervised contrastive item-space stack:

- development OOF Macro AP: **`0.5662217062664492`**;
- audit baseline: `0.5315527708634168`;
- delta vs audit baseline: `+0.0346689354030324`;
- category-specialist base: `0.5476780661335778`;
- delta vs category base: **`+0.018543640132871353`**;
- raw supervised semantic cosine Macro AP: `0.40597111640267125`;
- held-fold AP: `0.5692965046798911`, `0.5683388560864314`, `0.5694312406050667`, `0.5632500994392833`, `0.5684466083884651`;
- **all five held folds improve** versus the category base;
- workflow `31483288887`, source `b30821f613bf7051da51c42b64c7f79361d5619c`;
- private aggregate `experiments/v5/contrastive-sprint/b30821f613bf/aggregate`;
- metrics artifact ID `9099713308`.

This is development evidence only. No v5 submission has been retained and sealed gold remains unopened.

## v5 immutable validation protocol

- authoritative human rows: `365,654`;
- connected item components: `345,654`;
- development rows: `285,210`;
- sealed gold rows: **`80,444`**;
- development folds: `5`;
- cross-split item overlap: **`0`**;
- split SHA-256: **`aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`**;
- sealed gold metric opened: **false**;
- sealed gold rows scored: **0**.

Rules:

1. Do not use sealed-gold labels for model choice, calibration, feature selection, hard-negative mining or blend weights.
2. During development, do not encode/mine sealed-gold items as unlabeled adaptation data.
3. Freeze candidate, preprocessing, config and artifact hashes before the one-shot gold evaluation.
4. The requested `0.60` target means honest development OOF on this immutable split, not a repeatedly tuned holdout score.

Validation audit: run `31479778679`, source `93e4396330997c41bfb309f449f1dcb79a5e4db6`, private `experiments/v5/validation/93e439633099`.

## Retained v5 development ladder

### 1. Category specialists — KEEP

- OOF `0.5476780661335778`;
- delta vs audit baseline `+0.016125295270161044`;
- all five folds improve;
- private `experiments/v5/category/e885961388d1`.

This is the retained structured base for later comparisons.

### 2. Leakage-safe weak category specialists — KEEP

- OOF **`0.5514237338676234`**;
- delta vs category base `+0.00374566773404561`;
- all five folds improve;
- run `31484641329`, source `319993a469cfa37770d66cfaf1b2203515dc9841`;
- private `experiments/v5/weak-specialists/319993a469cf/aggregate`;
- weak input `11,187,780`, presample `250,000`, final weak rows `150,000` per fold;
- held-fold and sealed-gold items excluded from weak sampling.

### 3. Cross-fitted category/weighted/pretrained combo — KEEP intermediate

- OOF `0.559512531439709`;
- delta vs category base `+0.011834465306131192`;
- every fold improves by roughly `+0.012`;
- run `31485240666`, source `7a1c1764a2bdda8f007b9bfea7d088911623e7f0`;
- private `experiments/v5/combo/7a1c1764a2bd`.

The second level is itself cross-fitted; no row is trained on its own target plus an in-sample base prediction.

### 4. Strict train-only sparse TF-IDF category specialists — KEEP

- OOF **`0.5651306838802859`**;
- delta vs category base **`+0.017452617746708032`**;
- held-fold AP: `0.5638679943688357`, `0.5644112795204157`, `0.5729250493264109`, `0.5646827428419179`, `0.5676889194024459`;
- all five folds improve;
- run `31485396599`, source `634ee66890c39ad97c0fa725135b1b00e56ac126`;
- metrics artifact ID `9099873750`.

Vocabulary/IDF are learned only on outer-train items; held items enter via `transform()` only. This is strong evidence that rare model/SKU tokens provide transferable signal.

### 5. Supervised contrastive item-space stack — CURRENT DEV BEST / KEEP

- OOF **`0.5662217062664492`**;
- delta vs category base **`+0.018543640132871353`**;
- raw semantic cosine AP `0.40597111640267125`;
- all five folds improve;
- run `31483288887`, source `b30821f613bf7051da51c42b64c7f79361d5619c`;
- private `experiments/v5/contrastive-sprint/b30821f613bf/aggregate`;
- metrics artifact ID `9099713308`.

Initial physical batch `96` OOMed on MPS. The successful run preserved effective batch `96` via microbatch `24` and gradient accumulation `4`. The OOM was an infrastructure event, not a model rejection.

## Rejected / diagnostic branches — do not repeat blindly

### Direct attribute log-likelihood score addition — REJECT

- OOF `0.523218903672764` vs audit base `0.5315527708634168`;
- delta `-0.008333867190652766`;
- all five folds regress.

Do not rescue it with a scalar tuned on the same folds. Attribute specificity may enter as ordinary estimator features instead.

### Fold-weighted category specialists — diagnostic OOF input only

- OOF `0.5498696731704964`;
- delta vs category base `+0.0021916070369185636`;
- folds 2 and 3 regress slightly.

Do not promote standalone; it may be used only as already-OOF input to another independently cross-fitted layer.

### Pretrained multilingual bi-encoder — insufficient standalone

- stacked OOF `0.5318080650341337`;
- delta vs audit base `+0.0002552941707169021`;
- raw semantic cosine AP about `0.3120`.

Ready-made embeddings alone are not enough; supervised task adaptation is what produced the material neural gain.

### First `ruBert-base` pair teacher — INTEGRATION FAIL, not model REJECT

Workflow `31485127564`, source `00cc48dca806752d92496fa79f703a9ce3bcce63`, failed on all five folds before any comparable OOF prediction was produced. Exact failure:

`TypeError: build_reranker_examples() missing 1 required positional argument: 'attribute_importance'`

`train_v5_teacher_fold.py` still called `build_reranker_examples(items, curriculum)` after the helper API required `(items, pairs, attribute_importance)`. Fix the integration call and add an integration-level contract test before rerunning. Never record a teacher AP or mark the modeling hypothesis rejected from this run.

## Active v5 branches at this snapshot

Do not invent metrics until aggregate completion:

- explicit per-key attribute specialists: run `31485990777` — in progress;
- field-aware weak ranking teacher: run `31486298300` — in progress.

## Debugging / implementation lessons

- >11M weak rows: keep deterministic PyArrow streaming/bounded sampling; never materialize the whole table in pandas before loading a Transformer.
- MPS physical batch 96 can exhaust memory; preserve the intended effective batch with microbatching/gradient accumulation instead of changing the statistical experiment silently.
- Train-only TF-IDF unit tests must verify unseen transform, symmetry and finite/bounded values, not force a hand-written OOV ranking outcome.
- An apparent serializer regression was actually the next intentionally RED embedding test; inspect the exact failing test before assigning blame.
- Weak sampling must exclude held-fold and sealed-gold items.
- Helper APIs used by heavy workflows need integration tests, not only unit tests around isolated helper functions; the first ruBERT teacher failure is the canonical example.

## Historical artifacts

v4 remains a reproducible historical package, not the production anchor:

- v4 local OOF routing `0.5276431099433088`;
- v4 full-fit coefficient score `0.5284493942551521`;
- canonical ZIP SHA-256 `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`;
- exact organizer-image offline smoke passed.

v3 canonical historical package: local `0.5254642645846543`, SHA `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`.

## Current action

1. Continue v5 only on split SHA `aae58f...eb55b`.
2. Current honest dev target gap: `0.6000000000 - 0.5662217063 = 0.0337782937`.
3. Do not move the split or open sealed gold to chase the target.
4. Keep v2 as production fallback until a frozen v5 candidate passes one-shot sealed gold plus organizer runtime/package verification.
5. Record KEEP/REJECT/FAIL evidence immediately, but checkpoint Memora only from a GREEN repository state.

## Persistent Memora — operational rules

- pinned upstream `bc64ff745a9b2c0e6245e0137654f041fba0c155`;
- local SQLite + TF-IDF only;
- LLM/external embeddings/graph/auto-capture disabled;
- public source-backed files are canonical; private DB mirrors them for semantic retrieval;
- private DB lives under `agent-memory/latest/` with immutable checkpoints under `agent-memory/checkpoints/`.

Two v5 Memora runs (`31481012401`, `31482891498`) failed **before ingest** because memory-doc commits occurred while the repository was intentionally RED during TDD. In `31482891498`, test collection failed on missing `v5_weighted_specialists`; later GREEN code did not retroactively create a checkpoint.

A second memory-audit issue was found on 2026-08-11: `scripts/memory_ingest.py` indexed PLAN/RESULTS and durable Markdown but omitted machine-readable `experiments/CURRENT.json` and `v*/SAFE_METRICS.json`. A regression test now requires both sources to be part of `canonical_sources()`.

**Rule:** never weaken the full-test gate. A retained memory snapshot is valid only after full tests + `memory_policy.py`, Memora ingest, SQLite integrity/secret scan, private HF checkpoint upload and remote verification all pass.
