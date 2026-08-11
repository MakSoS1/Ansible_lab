# E-CUP Matching — Iteration v5 Results

Date: 2026-08-11
Status: **in progress — sealed gold unopened; no v5 submission retained**

## 1. Why v5 exists

Observed hidden Macro AP from submitted v1-v4 candidates:

| Candidate | Hidden Macro AP |
|---|---:|
| v1 | 0.23458522924335687 |
| v2 | **0.2583231811423486** |
| v3 non-canonical | 0.2583231811423486 |
| v3 canonical | 0.24810151893254498 |
| v4 canonical | 0.2531285194869718 |

**Operational conclusion:** v2 remains the production/leaderboard fallback. v5 must be selected on a new immutable protocol; old v3/v4 local scores are historical only.

## 2. Immutable v5 validation

Audit workflow `31479778679`, source `93e4396330997c41bfb309f449f1dcb79a5e4db6`, private prefix `experiments/v5/validation/93e439633099`.

- human rows: `365,654`;
- connected item components: `345,654`;
- development rows: `285,210`;
- sealed gold rows: `80,444`;
- development folds: `5`;
- cross-split item overlap: `0`;
- split SHA-256: **`aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`**;
- gold metric opened: **false**;
- gold rows scored: **0**.

Audit baseline OOF Macro AP: **`0.5315527708634168`**.

Held-fold baseline AP:

1. `0.535137201283413`
2. `0.5293875046954674`
3. `0.5375146978687112`
4. `0.531512424671052`
5. `0.5337619678140229`

Fold std: `0.0028186164376135885`. The requested `0.60` target must be reached on this same protocol, not by changing the split or repeatedly opening gold.

## 3. Category-specialist structured base — KEEP

Canonical downstream OOF: **`0.5476780661335778`**.
Delta vs audit baseline: `+0.016125295270161044`.
Private prefix: `experiments/v5/category/e885961388d1`.

Held-fold AP:

1. `0.5504494811533941`
2. `0.5478362369858445`
3. `0.554178032571296`
4. `0.5453862553120211`
5. `0.5500737968478944`

All five folds improve. This is the retained structured base against which later v5 signals are judged.

An earlier near-identical safe snapshot reported `0.5476778075943867`; downstream workflows use the canonical `0.5476780661335778` OOF file. Do not treat the tiny difference as a separate architecture.

## 4. Direct attribute likelihood score addition — REJECT

Workflow `31482167758`, source `f19e58475dd877dc1e1c27b4671b26b8e981d902`, private `experiments/v5/attribute/f19e58475dd8`.

- candidate OOF: `0.523218903672764`;
- delta vs audit baseline: `-0.008333867190652766`;
- every fold regressed.

Fold deltas: `-0.0044078029`, `-0.0062674337`, `-0.0108360696`, `-0.0094291089`, `-0.0092964004`.

**Decision:** REJECT direct logit addition. Do not tune a rescue scalar on these same folds. Attribute-key specificity may be passed as ordinary estimator features instead.

## 5. Pretrained multilingual bi-encoder — insufficient standalone

Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

- stacked OOF: `0.5318080650341337`;
- delta vs audit base: `+0.0002552941707169021`;
- raw semantic cosine Macro AP: `0.31201359969464276`;
- gold items encoded: `0`.

The small positive gain across folds is not enough to retain standalone. This branch established that ready-made embeddings alone are weak; task supervision is required.

## 6. Fold-weighted category specialists — diagnostic OOF input only

Workflow `31483353777`, source `9df24f7ee1335046d00751d4c2b0cbd78b00e162`, private aggregate `experiments/v5/weighted-specialists/9df24f7ee133/aggregate`, artifact `9098324849`, artifact digest `ec9614c9906cf4b9c39c613a96c73294d885158d0bd3ec358b911600ca13cea9`.

- OOF: `0.5498696731704964`;
- delta vs category base: `+0.0021916070369185636`.

Fold deltas vs category base:

- `+0.0021354161959221685`
- `+0.004859795667646272`
- `-0.00031919882576014746`
- `-0.0009191637590409973`
- `+0.004585239432923327`

**Decision:** not retained standalone because folds 2 and 3 regress. Keep only as an already-OOF candidate input to a separately cross-fitted stack.

## 7. Leakage-safe weak category specialists — KEEP

Workflow **`31484641329`**, source **`319993a469cfa37770d66cfaf1b2203515dc9841`**, private aggregate `experiments/v5/weak-specialists/319993a469cf/aggregate`, artifact `9099098118`, digest `3f8b2db519f4f6925a8cc6f7bffb7cde7869d09c50a12e5be957c0894089ee93`.

Curriculum invariants:

- weak input: `11,187,780` rows;
- confidence-filtered presample: `250,000`;
- final weak rows: `150,000` per fold;
- held-fold items excluded: **true**;
- sealed-gold items excluded: **true**;
- gold rows used/scored: `0`.

Result:

- OOF: **`0.5514237338676234`**;
- delta vs category base: **`+0.00374566773404561`**;
- every fold improves.

| Fold | Category base | Weak specialist | Delta |
|---:|---:|---:|---:|
| 0 | 0.5504494811533941 | 0.5546298909754048 | +0.004180409822010733 |
| 1 | 0.5478362369858445 | 0.5520443319510406 | +0.004208094965196141 |
| 2 | 0.554178032571296 | 0.5570904113247204 | +0.002912378753424316 |
| 3 | 0.5453862553120211 | 0.5470536921958663 | +0.0016674368838451858 |
| 4 | 0.5500737968478944 | 0.5551003946611213 | +0.005026597813226896 |

**Decision:** KEEP. Weak supervision transfers when held/gold items are explicitly forbidden from the curriculum.

## 8. Cross-fitted category + weighted + pretrained combo — KEEP intermediate

Workflow **`31485240666`**, source **`7a1c1764a2bdda8f007b9bfea7d088911623e7f0`**, private `experiments/v5/combo/7a1c1764a2bd`, artifact `9098856613`, digest `b020be418c698384c45f0b3bf9c3a5acfb1e0d66eea79ad5f2caf4930dc86ede`.

All base inputs are OOF and the meta-estimator is itself cross-fitted.

- OOF: **`0.559512531439709`**;
- delta vs category base: **`+0.011834465306131192`**;
- delta vs audit base: `+0.0279597605762922`;
- gold unopened.

Held-fold AP: `0.562580065789817`, `0.5605549739646596`, `0.5667063354890245`, `0.5579194708823281`, `0.5631351691480011`.

All five folds improve by about `+0.012`. **KEEP intermediate**, but it has since been surpassed by sparse and supervised contrastive branches.

## 9. Strict train-only sparse TF-IDF category specialists — KEEP

Workflow **`31485396599`**, source **`634ee66890c39ad97c0fa725135b1b00e56ac126`**, metrics artifact **`9099873750`**, artifact digest `18ed09639b0987625ba83ab10b63393ee022f6a34aff1cd051f3cc7751a2f8dd`.

Vocabulary and IDF are fitted only on outer-train items. Held items are unseen and enter only through `transform()`.

- OOF: **`0.5651306838802859`**;
- delta vs category base: **`+0.017452617746708032`**;
- gold opened: false; gold rows scored: 0.

Held-fold AP / delta:

| Fold | Sparse AP | Delta vs category base |
|---:|---:|---:|
| 0 | 0.5638679943688357 | +0.01341851321544163 |
| 1 | 0.5644112795204157 | +0.016575042534571205 |
| 2 | 0.5729250493264109 | +0.01874701675511481 |
| 3 | 0.5646827428419179 | +0.019296487529896766 |
| 4 | 0.5676889194024459 | +0.017615122554551554 |

All five folds improve. **KEEP.** Rare SKU/model-code weighting is a strong independent signal and should only be combined through leakage-safe OOF methods.

## 10. Supervised contrastive item-space stack — CURRENT DEVELOPMENT BEST / KEEP

Workflow **`31483288887`**, source **`b30821f613bf7051da51c42b64c7f79361d5619c`**, private aggregate **`experiments/v5/contrastive-sprint/b30821f613bf/aggregate`**, metrics artifact **`9099713308`**, artifact digest `79615471ea6567625896d24c8e95cc5034958c90b21ff39a645329729acc1079`.

The first launch OOMed on MPS with physical batch `96` before meaningful training evidence. The corrected run preserves effective batch `96` using physical microbatch `24` and gradient accumulation `4`.

Final successful outer-CV result:

- category base: `0.5476780661335778`;
- **stacked OOF: `0.5662217062664492`**;
- **delta vs category base: `+0.018543640132871353`**;
- delta vs audit baseline: `+0.0346689354030324`;
- raw supervised semantic cosine Macro AP: **`0.40597111640267125`**;
- gold opened: false; gold rows scored: 0.

Held-fold AP / delta:

| Fold | Contrastive stack | Delta vs category base | Raw semantic cosine AP |
|---:|---:|---:|---:|
| 0 | 0.5692965046798911 | +0.01884702352649703 | 0.4139705522068841 |
| 1 | 0.5683388560864314 | +0.020502619100586927 | 0.4118216953871334 |
| 2 | 0.5694312406050667 | +0.015253208033770727 | 0.4058547870713725 |
| 3 | 0.5632500994392833 | +0.017863844127262163 | 0.39547716343930234 |
| 4 | 0.5684466083884651 | +0.01837281154057069 | 0.4034094420314965 |

All five folds improve. **KEEP and use `0.5662217062664492` as the current honest development benchmark.**

Important inference from the ablation pair:

- pretrained item-space stack: `0.5318080650`;
- supervised item-space stack: `0.5662217063`.

Therefore task supervision, not merely a stronger pretrained embedding, is responsible for the meaningful semantic gain.

## 11. First `ruBert-base` pair teacher — INTEGRATION FAIL BEFORE METRICS

Workflow **`31485127564`**, source **`00cc48dca806752d92496fa79f703a9ce3bcce63`**.

All five fold jobs failed before held-fold predictions; aggregate was skipped. Exact error:

`TypeError: build_reranker_examples() missing 1 required positional argument: 'attribute_importance'`

The runner called `build_reranker_examples(items, curriculum)` after the helper API required `(items, pairs, attribute_importance)`.

**Decision:** this run is neither KEEP nor REJECT. It is a code/integration failure. Fix the call explicitly and add an integration-level test covering the composed runner path before rerunning. Never fabricate a ruBERT score from this failure.

## 12. Still-active experiments at this snapshot

- explicit per-key attribute specialists: run `31485990777`, source `cb350b4e7ba6bb4a6d283f91bae4d6ea13235d57` — **in progress**;
- field-aware weak ranking teacher: run `31486298300`, source `411a5349fe731506757fdc1a3c8857a370225fb8` — **in progress**.

Do not claim metrics until each has a completed aggregate.

## 13. Implemented foundations that must not be lost

- deterministic balanced component-disjoint split manifest + hash;
- hard gold eligibility checks and pre-evaluation freeze hashes;
- paired component bootstrap;
- v2-anchored residual correction primitive;
- identity-first item serializer preserving name/brand/model/numeric/attribute sections;
- compact symmetric dense-embedding pair features;
- train-only sparse TF-IDF with unseen-item `transform`;
- strict OOF second-level stacking;
- leakage-safe OOF hard-negative curriculum;
- source/sample weighting for human vs weak rows;
- explicit-key attribute and field-aware ranking-teacher experimental paths.

## 14. Debugging lessons — do not repeat

1. **Do not reuse the old 73,131-row holdout as the v5 tuning loop.** Hidden evidence showed local improvements did not transfer monotonically.
2. **Do not open sealed gold while developing.** Gold is one-shot after candidate/config/preprocessing freeze.
3. **Do not materialize all >11M weak rows with pandas before a Transformer.** Use bounded PyArrow streaming/two-pass sampling and finish CPU-heavy preparation before loading the large model.
4. **Do not interpret MPS/RTX resource failures as model scores.** Diagnose resource root cause first.
5. **TF-IDF tests should validate representation contracts, not hand-written desired rankings for OOV examples.** OOF decides utility.
6. **Do not directly add learned attribute likelihood to logits.** It regressed every fold.
7. **Do not blame the most recent implementation without reading the exact failing test.** A serializer commit looked broken because the next deliberately RED embedding test was already present.
8. **Heavy workflow integration needs integration tests.** The first ruBERT teacher passed isolated tests but failed the real helper call signature.
9. **Weak rows must exclude held-fold and sealed-gold items.** This invariant is required, not optional.
10. **Memora checkpoint only from GREEN state.** Earlier v5 memory runs failed before ingest while TDD was intentionally RED; later GREEN commits do not retroactively create missed checkpoints.
11. **Machine-readable state must be ingested too.** `CURRENT.json` and `v*/SAFE_METRICS.json` are now required Memora canonical sources, not merely files sitting in Git.

## 15. Current state

- production/hidden fallback: **v2**, hidden `0.2583231811423486`;
- audit baseline: `0.5315527708634168` OOF;
- category structured base: `0.5476780661335778`;
- weak specialists: `0.5514237338676234`;
- cross-fitted combo: `0.559512531439709`;
- sparse specialists: `0.5651306838802859`;
- **current development best: supervised contrastive `0.5662217062664492`**;
- stretch target: `0.6000000000`, remaining gap `0.0337782937335508`;
- sealed gold: **80,444 rows, unopened, 0 scored**;
- v5 submission: **not retained**.

Continue from this state. Never regress the validation protocol to make the metric easier, and never promote a v5 submission before the one-shot gold plus organizer runtime/package gates.
