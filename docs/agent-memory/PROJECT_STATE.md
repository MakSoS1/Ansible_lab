# E-CUP Matching — Canonical Project State

Updated: 2026-08-11

## Objective

Score strongly in the ODS E-CUP 2026 Ozon pairwise product matching task while keeping model selection honest for unseen products, the final submission reproducible/offline-compatible, and all competition artifacts private.

## Fixed task/evaluation facts

- 20 product categories.
- Official metric: unweighted macro mean of `sklearn.metrics.average_precision_score` across categories.
- Human labels: `365,654` pairs; soft LLM labels: `>11M` pairs.
- Hidden test contains new/unseen products; item leakage is unacceptable.
- Submission CLI: `--items_path`, `--matches_path`, `--output_path`; output `id1,id2,predict`.
- Organizer image: `odsai/ecup26-matching-baseline:1.0`.
- Public Git contains code/docs only. Raw data, learned models, OOF prediction files, Memora DBs and submission ZIPs remain private.
- Private artifact/data repo: `Maksim123321/e-cup-2026-matching-private`.

## Two different meanings of “best” — never merge them

### Production / hidden-leaderboard anchor

The best observed hidden score among submitted v1-v4 candidates is **v2**:

| Submission | Hidden Macro AP |
|---|---:|
| v1 | 0.23458522924335687 |
| v2 | **0.2583231811423486** |
| v3 non-canonical | 0.2583231811423486 |
| v3 canonical | 0.24810151893254498 |
| v4 canonical | 0.2531285194869718 |

Therefore **v2 is the immutable production/leaderboard fallback while v5 is being developed**. Old v3/v4 local gains remain valid historical offline measurements but did not transfer monotonically to hidden evaluation and are no longer the primary model-selection proxy.

### Current development best

v5 is **in progress**. The strongest honest development OOF result currently available is the leakage-safe cross-fitted combo:

- development OOF Macro AP: **`0.559512531439709`**;
- baseline OOF: `0.5315527708634168`;
- delta vs baseline: `+0.0279597605762922`;
- category-specialist base: `0.5476780661335778`;
- delta vs category base: `+0.011834465306131192`;
- held-fold AP: `0.562580065789817`, `0.5605549739646596`, `0.5667063354890245`, `0.5579194708823281`, `0.5631351691480011`;
- **all five held folds improved** relative to the category-specialist base;
- run `31485240666`, source `7a1c1764a2bdda8f007b9bfea7d088911623e7f0`;
- private prefix `experiments/v5/combo/7a1c1764a2bd`.

This is development evidence only. No v5 submission is retained yet.

## v5 immutable validation protocol

v5 replaced the repeatedly reused 73,131-row development holdout with a new sealed protocol before further model selection:

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
2. During development, do not encode/mine sealed-gold items as if they were unlabeled adaptation data.
3. Candidate, preprocessing, config and relevant artifact hashes must be frozen before the one-shot gold evaluation.
4. The user-requested `0.60` target means honest development OOF on this immutable split, not a repeatedly tuned holdout score.

Validation-audit run: `31479778679`, source `93e4396330997c41bfb309f449f1dcb79a5e4db6`, private prefix `experiments/v5/validation/93e439633099`.

## v5 retained development steps

### Category specialists — KEEP

Train one structured HGB per category instead of sharing one global HGB across heterogeneous product regimes.

- exact current category-base OOF: **`0.5476780661335778`**;
- delta vs audit baseline: `+0.016125295270161044`;
- all five folds improved;
- private prefix `experiments/v5/category/e885961388d1`.

This is the retained structured base for later v5 comparisons.

### Leakage-safe weak category specialists — KEEP

Weak LLM rows are filtered so neither held-fold items nor sealed-gold items enter the fold curriculum.

- category base: `0.5476780661335778`;
- weak-specialist OOF: **`0.5514237338676234`**;
- delta: **`+0.00374566773404561`**;
- every fold improved: `+0.0041804098`, `+0.0042080950`, `+0.0029123788`, `+0.0016674369`, `+0.0050265978`;
- run `31484641329`, source `319993a469cfa37770d66cfaf1b2203515dc9841`;
- private prefix `experiments/v5/weak-specialists/319993a469cf/aggregate`;
- sprint used `250,000` confidence-filtered presample rows and `150,000` final weak rows per fold from `11,187,780` weak input rows.

### Cross-fitted combo — current development best

Cross-fit combines already-OOF category-specialist, fold-weighted and pretrained semantic signals. The second level never trains on in-sample base predictions of its own held fold.

- OOF: **`0.559512531439709`**;
- all five folds improved vs category base;
- private prefix `experiments/v5/combo/7a1c1764a2bd`.

KEEP as the current development benchmark, but **not yet a retained submission**.

## v5 rejected / diagnostic branches — do not repeat blindly

### Direct attribute log-likelihood score addition — REJECT

- OOF `0.523218903672764` vs base `0.5315527708634168`;
- delta `-0.008333867190652766`;
- all five folds regressed.

Do not rescue this branch by tuning a scalar on the same development folds. Attribute information may be supplied as ordinary model features instead of a direct logit shift.

### Fold-weighted category specialists — diagnostic input only

- OOF `0.5498696731704964`;
- delta vs category base `+0.0021916070369185636`;
- folds 2 and 3 regressed slightly (`-0.0003191988`, `-0.0009191638`).

Do not promote standalone. It is allowed as an OOF input to a separately cross-fitted stack because the combo proved consistent gain.

### Pretrained multilingual bi-encoder without supervised adaptation — insufficient standalone

- stacked OOF `0.5318080650341337`;
- delta vs audit base only `+0.0002552941707169021`;
- raw semantic cosine Macro AP about `0.3120`.

A ready-made item embedding space is not enough; a neural path must prove supervised/weak OOF value over the retained structured base.

## Active v5 branches at this snapshot

Do not invent metrics for these until their aggregate run completes:

- supervised contrastive outer-CV: run `31483288887` — in progress;
- `ruBert-base` teacher outer-CV: run `31485127564` — in progress;
- strict train-only sparse specialists: run `31485396599` — in progress;
- explicit per-attribute specialists: run `31485990777` — in progress;
- field-aware weak ranking teacher: run `31486298300` — queued at this snapshot.

## v5 debugging / implementation lessons

- MPS contrastive physical batch `96` caused OOM at roughly the full available MPS memory. The fix preserves effective batch `96` using physical microbatch `24` plus gradient accumulation `4`; do not interpret the OOM as a negative ML result.
- Train-only TF-IDF tests must verify representation contracts (unseen `transform`, symmetry, finite/bounded values), not force a hand-written ranking such as unknown SKU/color conflicts. OOF decides utility.
- An apparent serializer CI failure was caused by the next intentionally RED embedding test in the same evolving branch, not the serializer. Always read the exact failing test/stack trace before attributing a regression.
- Weak sampling must exclude held-fold and sealed-gold items. The successful weak-specialist run asserts these invariants.
- For >11M weak rows, keep PyArrow streaming/bounded sampling; do not return to full-table pandas materialization before loading a Transformer.

## Historical v4/v3 artifacts

v4 remains a reproducible historical package, not the current production anchor:

- v4 OOF routing score `0.5276431099433088`;
- v4 full-fit coefficient score `0.5284493942551521`;
- canonical ZIP SHA-256 `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`;
- exact organizer-image offline smoke passed.

v3 canonical historical package:

- local Macro AP `0.5254642645846543`;
- canonical SHA-256 `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`.

Do not reinterpret these old local metrics as evidence that v3/v4 beat v2 on hidden evaluation.

## Current action

1. Continue v5 only on the immutable split SHA `aae58f...eb55b`.
2. Compare new branches against current OOF benchmarks; do not move the split or open gold to chase `0.60`.
3. Keep v2 as production/leaderboard fallback until a frozen v5 candidate passes the one-shot sealed-gold gate plus organizer runtime/package verification.
4. Update durable memory after each meaningful KEEP/REJECT/FAIL result, but checkpoint Memora only from a GREEN repository state.

## Persistent agent memory — operational

Hardened Memora remains pinned/local-only:

- upstream `bc64ff745a9b2c0e6245e0137654f041fba0c155`;
- SQLite + TF-IDF only;
- LLM/external embeddings/graph/auto-capture disabled;
- public Markdown is canonical;
- private DB lives under `agent-memory/latest/` with immutable checkpoints under `agent-memory/checkpoints/`.

Important incident: two v5 Memora runs (`31481012401`, `31482891498`) failed **before ingest** because their memory-doc commits occurred while the repository was intentionally RED during TDD. In run `31482891498`, collection failed with `ModuleNotFoundError: ecup_matching.ml.v5_weighted_specialists`; subsequent production code made the workspace GREEN, but that did not retroactively create the skipped checkpoint.

**Rule:** never weaken the test gate. A retained memory snapshot is valid only after full tests + `memory_policy.py` pass, Memora ingestion succeeds, SQLite integrity/secret checks pass, and the private checkpoint is uploaded/verified. If a docs-triggered run occurred during RED TDD, force a later memory-triggering commit or manual dispatch after GREEN.
