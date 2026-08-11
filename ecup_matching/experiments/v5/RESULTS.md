# E-CUP Matching — Iteration v5 Results

Date: 2026-08-11
Status: **in progress — sealed gold unopened; no v5 submission retained**

## Validation contract

v5 development uses one immutable component-disjoint protocol:

- human rows `365,654`;
- connected item components `345,654`;
- development rows `285,210`;
- sealed gold rows `80,444`;
- 5 dev folds;
- cross-split item overlap `0`;
- split SHA-256 **`aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`**;
- gold metric opened **false**; gold rows scored **0**.

Audit workflow `31479778679`, source `93e4396330997c41bfb309f449f1dcb79a5e4db6`, private `experiments/v5/validation/93e439633099`.

Audit baseline OOF: **`0.5315527708634168`**. Fold AP: `0.535137201283413`, `0.5293875046954674`, `0.5375146978687112`, `0.531512424671052`, `0.5337619678140229`; std `0.0028186164376135885`.

The user-requested `0.60` target is judged on this exact development protocol. Never change the split or open sealed gold to make the number easier.

## Production anchor versus development best

Hidden scores from submitted v1-v4 candidates: v1 `0.23458522924335687`, v2 **`0.2583231811423486`**, v3 noncanonical `0.2583231811423486`, v3 canonical `0.24810151893254498`, v4 canonical `0.2531285194869718`.

Therefore **v2 is the production/hidden fallback**, while v5 development-best is selected by the new honest OOF. These are separate concepts.

## Completed v5 ladder

### Category-specialist HGB — KEEP

Canonical OOF **`0.5476780661335778`**, +`0.016125295270161044` vs audit. Fold AP: `0.5504494811533941`, `0.5478362369858445`, `0.554178032571296`, `0.5453862553120211`, `0.5500737968478944`. All folds improve. Private `experiments/v5/category/e885961388d1`.

### Direct attribute likelihood score addition — REJECT

Run `31482167758`, source `f19e58475dd877dc1e1c27b4671b26b8e981d902`. OOF `0.523218903672764`, -`0.008333867190652766` vs audit; every fold regresses. Never rescue via a scalar tuned on the same folds. Key-specific information may be estimator features, not an unconditional logit shift.

### Pretrained multilingual bi-encoder — insufficient standalone

Stacked OOF `0.5318080650341337`, only +`0.0002552941707169021`; raw semantic cosine `0.31201359969464276`. Pretrained item-space alone is weak; later supervised item-space is materially better.

### Fold-weighted category specialists — diagnostic OOF input only

Run `31483353777`, source `9df24f7ee1335046d00751d4c2b0cbd78b00e162`, artifact `9098324849`. OOF `0.5498696731704964`, +`0.0021916070369185636`; folds 2 and 3 regress slightly. Not retained standalone, but allowed as already-OOF input to a separate cross-fitted layer.

### Leakage-safe weak category specialists — KEEP

Run **`31484641329`**, source **`319993a469cfa37770d66cfaf1b2203515dc9841`**, private `experiments/v5/weak-specialists/319993a469cf/aggregate`, artifact `9099098118`.

- weak input `11,187,780`;
- presample `250,000`;
- final weak rows `150,000` per fold;
- held-fold and sealed-gold items excluded;
- OOF **`0.5514237338676234`**, +`0.00374566773404561` vs category base;
- all five folds improve.

### Cross-fitted category + weighted + pretrained combo — KEEP intermediate

Run **`31485240666`**, source **`7a1c1764a2bdda8f007b9bfea7d088911623e7f0`**, private `experiments/v5/combo/7a1c1764a2bd`, artifact `9098856613`.

OOF **`0.559512531439709`**, +`0.011834465306131192` vs category base. Fold AP `0.562580065789817`, `0.5605549739646596`, `0.5667063354890245`, `0.5579194708823281`, `0.5631351691480011`; all improve. Every input is OOF and the second level is also cross-fitted.

### Strict train-only sparse TF-IDF specialists — KEEP

Run **`31485396599`**, source **`634ee66890c39ad97c0fa725135b1b00e56ac126`**, artifact `9099873750`, artifact digest `18ed09639b0987625ba83ab10b63393ee022f6a34aff1cd051f3cc7751a2f8dd`.

Vocabulary/IDF are fit only on outer-train items; held items enter only through `transform()`.

- OOF **`0.5651306838802859`**;
- delta vs category base **`+0.017452617746708032`**;
- folds `0.5638679943688357`, `0.5644112795204157`, `0.5729250493264109`, `0.5646827428419179`, `0.5676889194024459`;
- all five improve.

Rare model/SKU tokens are a strong independent transferable signal.

### Supervised contrastive item-space stack — KEEP

Run **`31483288887`**, source **`b30821f613bf7051da51c42b64c7f79361d5619c`**, private `experiments/v5/contrastive-sprint/b30821f613bf/aggregate`, artifact `9099713308`, artifact digest `79615471ea6567625896d24c8e95cc5034958c90b21ff39a645329729acc1079`.

Initial physical batch 96 OOMed on MPS before meaningful evidence. Successful run preserves effective batch 96 with microbatch 24 + gradient accumulation 4.

- OOF **`0.5662217062664492`**;
- delta vs category base **`+0.018543640132871353`**;
- raw semantic cosine AP `0.40597111640267125`;
- folds `0.5692965046798911`, `0.5683388560864314`, `0.5694312406050667`, `0.5632500994392833`, `0.5684466083884651`;
- all five improve.

The contrast with pretrained OOF `0.5318080650` shows that **task supervision**, not merely a pretrained embedding model, creates the useful semantic representation.

### Explicit per-key attribute category specialists — CURRENT DEV BEST / KEEP

Run **`31485990777`**, source **`cb350b4e7ba6bb4a6d283f91bae4d6ea13235d57`**, metrics artifact **`9100228112`**, artifact digest `6417c94041c3443f03acf85227dceb94e65abea668d1b33bc6dc477f41f5a8fb`.

This branch is deliberately different from rejected direct attribute likelihood. It supplies explicit fold-trained per-key equal/conflict/missing features to each category estimator, so the model itself decides when key identity matters.

- category base `0.5476780661335778`;
- **OOF `0.5683065131240066`**;
- **delta `+0.02062844699042876`**;
- delta vs audit baseline `+0.036753742260589806`;
- sealed gold unopened / 0 rows scored.

Fold AP / delta vs category base:

| Fold | AP | Delta |
|---:|---:|---:|
| 0 | 0.5706378464826163 | +0.02018836532922219 |
| 1 | 0.5682631251392076 | +0.020426888153363132 |
| 2 | 0.5754313094571646 | +0.021253276885868533 |
| 3 | 0.5633705139683869 | +0.01798425865636577 |
| 4 | 0.5731185912680369 | +0.02304479442014251 |

All five folds improve. Per-category OOF includes notable weak-category gains such as Обувь `0.3709070740115228`, Одежда `0.37773930906821385`, Мебель `0.48050849118772493`, Электроника `0.46493466586646304`, Ювелирные изделия `0.40465501825181127`.

**Decision:** current honest development benchmark becomes **`0.5683065131240066`**.

## Failed heavy run that must not be misremembered

### First `ruBert-base` pair teacher — INTEGRATION FAIL, not model REJECT

Workflow **`31485127564`**, source **`00cc48dca806752d92496fa79f703a9ce3bcce63`**. All five folds failed before comparable predictions; aggregate skipped.

Exact error: `TypeError: build_reranker_examples() missing 1 required positional argument: 'attribute_importance'`.

The runner still called `build_reranker_examples(items, curriculum)` after the helper signature required `(items, pairs, attribute_importance)`. Fix the integration call and cover the composed path with an integration-level test before rerunning. Never claim a ruBERT AP from this run.

## Still running at this snapshot

- field-aware weak ranking teacher: run `31486298300`, source `411a5349fe731506757fdc1a3c8857a370225fb8` — **in progress**.

Do not invent or infer its metric before aggregate completion.

## Implementation/debugging lessons to preserve

1. Do not return to the old 73,131-row holdout as v5's primary tuning loop.
2. Do not open sealed gold while developing; it is one-shot post-freeze verification.
3. For >11M weak rows, use deterministic bounded PyArrow streaming; never materialize the full table in pandas before loading a Transformer.
4. Infrastructure failures (MPS OOM, RTX exit 137) are not model scores.
5. TF-IDF unit tests validate unseen transform/symmetry/finite bounds, not a hand-written OOV ranking.
6. Direct attribute likelihood shifts are harmful, while explicit per-key estimator features are beneficial; do not conflate them.
7. Read exact failing tests before blaming the latest implementation; overlapping RED TDD cycles previously made serializer code look guilty when the missing next module was the actual failure.
8. Weak sampling must forbid held-fold and sealed-gold items.
9. Heavy workflow helper composition requires integration tests; first ruBERT teacher is the canonical stale-API example.
10. Memora checkpoints must come only from GREEN repository state.
11. `experiments/CURRENT.json` and `v*/SAFE_METRICS.json` are first-class Memora sources, not files that may sit outside semantic indexing.

## Memora audit incident

Earlier v5 Memora runs `31481012401` and `31482891498` failed **before ingest** because their memory-triggering commits landed during intentionally RED TDD. In `31482891498`, collection failed with missing `ecup_matching.ml.v5_weighted_specialists`; later GREEN code did not retroactively create the skipped checkpoint.

A second audit found `scripts/memory_ingest.py` did not include machine-readable `CURRENT.json` or `SAFE_METRICS.json`. A regression test now requires both in `canonical_sources()`; do not weaken the test or the full GREEN checkpoint gate.

## Current state

- production fallback: **v2**, hidden `0.2583231811423486`;
- audit baseline `0.5315527708634168`;
- category base `0.5476780661335778`;
- weak `0.5514237338676234`;
- combo `0.559512531439709`;
- sparse `0.5651306838802859`;
- supervised contrastive `0.5662217062664492`;
- **current dev best explicit attributes `0.5683065131240066`**;
- stretch target `0.60`; remaining gap `0.0316934868759934`;
- sealed gold `80,444`, unopened, 0 scored;
- v5 submission not retained.

Continue on the immutable split. Promote no v5 submission before candidate/config/preprocessing freeze, one-shot sealed-gold evaluation, and organizer runtime/package gates.
