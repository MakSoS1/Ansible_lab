# E-CUP Matching — Canonical Project State

Updated: 2026-08-11

## Objective

Score strongly in ODS E-CUP 2026 Ozon product matching while keeping model selection honest for unseen products, final submission reproducible/offline-compatible, and competition artifacts private.

## Fixed facts

- 20 categories; official metric is unweighted category Macro Average Precision.
- Human labels: `365,654`; soft LLM labels: `>11M`.
- Hidden test contains unseen products; item leakage is unacceptable.
- Organizer image: `odsai/ecup26-matching-baseline:1.0`.
- Public Git contains code/source-backed docs only. Raw data, models, OOF predictions, Memora DBs and submission ZIPs stay private.
- Private artifact/data repo: `Maksim123321/e-cup-2026-matching-private`.

## Never merge these two meanings of “best”

### Production / hidden anchor

Observed hidden Macro AP: v1 `0.23458522924335687`, v2 **`0.2583231811423486`**, v3 noncanonical `0.2583231811423486`, v3 canonical `0.24810151893254498`, v4 canonical `0.2531285194869718`.

**v2 remains the immutable production/leaderboard fallback while v5 is developed.** Old v3/v4 local gains are historical and did not transfer monotonically to hidden evaluation.

### Current development best

Current honest v5 development best is **explicit per-key attribute category specialists**:

- OOF Macro AP: **`0.5683065131240066`**;
- audit baseline: `0.5315527708634168`;
- delta vs audit: `+0.036753742260589806`;
- category base: `0.5476780661335778`;
- delta vs category base: **`+0.02062844699042876`**;
- held-fold AP: `0.5706378464826163`, `0.5682631251392076`, `0.5754313094571646`, `0.5633705139683869`, `0.5731185912680369`;
- all five folds improve;
- run `31485990777`, source `cb350b4e7ba6bb4a6d283f91bae4d6ea13235d57`;
- metrics artifact ID `9100228112`, artifact digest `6417c94041c3443f03acf85227dceb94e65abea668d1b33bc6dc477f41f5a8fb`.

This is development evidence only. No v5 submission is retained and sealed gold remains unopened.

## Immutable v5 validation

- human rows `365,654`;
- item components `345,654`;
- development rows `285,210`;
- sealed gold rows **`80,444`**;
- five development folds;
- cross-split item overlap **`0`**;
- split SHA-256 **`aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`**;
- gold metric opened **false**; gold rows scored **0**.

Rules:

1. Never use sealed-gold labels for model choice/calibration/features/mining/blend weights.
2. Do not use sealed-gold items for representation adaptation or hard/weak mining during development.
3. Freeze candidate/config/preprocessing/artifact hashes before one-shot gold.
4. `0.60` means honest OOF on this immutable development protocol, not a repeatedly tuned holdout.

Audit: run `31479778679`, source `93e4396330997c41bfb309f449f1dcb79a5e4db6`, private `experiments/v5/validation/93e439633099`.

## Retained v5 development ladder

### Category specialists — KEEP

OOF **`0.5476780661335778`**, delta `+0.016125295270161044` vs audit, all folds improve. Private `experiments/v5/category/e885961388d1`. This remains the structured base used for incremental comparisons.

### Leakage-safe weak category specialists — KEEP

OOF **`0.5514237338676234`**, delta `+0.00374566773404561` vs category base, all folds improve. Run `31484641329`, source `319993a469cfa37770d66cfaf1b2203515dc9841`, private `experiments/v5/weak-specialists/319993a469cf/aggregate`. Weak input `11,187,780`; presample `250,000`; final weak `150,000` per fold; held and gold items excluded.

### Cross-fitted category + weighted + pretrained combo — KEEP intermediate

OOF **`0.559512531439709`**, delta `+0.011834465306131192` vs category base, all folds improve by about `+0.012`. Run `31485240666`, source `7a1c1764a2bdda8f007b9bfea7d088911623e7f0`, private `experiments/v5/combo/7a1c1764a2bd`. The meta layer is itself cross-fitted.

### Strict train-only sparse TF-IDF specialists — KEEP

OOF **`0.5651306838802859`**, delta `+0.017452617746708032` vs category base, all folds improve. Run `31485396599`, source `634ee66890c39ad97c0fa725135b1b00e56ac126`, artifact `9099873750`. Vocabulary/IDF fit only on outer-train items; held items only `transform()`.

### Supervised contrastive item-space stack — KEEP

OOF **`0.5662217062664492`**, delta `+0.018543640132871353` vs category base, raw semantic cosine `0.40597111640267125`, all folds improve. Run `31483288887`, source `b30821f613bf7051da51c42b64c7f79361d5619c`, private `experiments/v5/contrastive-sprint/b30821f613bf/aggregate`, artifact `9099713308`.

Initial physical batch 96 caused MPS OOM; successful run preserved effective batch 96 via microbatch 24 + accumulation 4. This was infrastructure failure, not model rejection.

### Explicit per-key attribute specialists — CURRENT DEV BEST / KEEP

Explicit fold-trained key features let each category model see separate match/conflict/missing information for selected attributes rather than only aggregate agreement/conflict ratios.

- OOF **`0.5683065131240066`**;
- delta vs category base **`+0.02062844699042876`**;
- folds: `0.5706378464826163`, `0.5682631251392076`, `0.5754313094571646`, `0.5633705139683869`, `0.5731185912680369`;
- fold deltas: `+0.02018836532922219`, `+0.020426888153363132`, `+0.021253276885868533`, `+0.01798425865636577`, `+0.02304479442014251`;
- all five folds improve;
- run `31485990777`, source `cb350b4e7ba6bb4a6d283f91bae4d6ea13235d57`, artifact `9100228112`.

This demonstrates that key identity (`size`, `memory`, `model`, etc.) contains useful information lost by aggregate attribute ratios.

## Rejected / diagnostic / failed branches

### Direct attribute likelihood shift — REJECT

OOF `0.523218903672764`, delta `-0.008333867190652766` vs audit, every fold regressed. Do not rescue with a scalar tuned on the same folds; key-specific evidence should enter as estimator features instead.

### Fold-weighted category specialists — diagnostic OOF input only

OOF `0.5498696731704964`, delta `+0.0021916070369185636`; folds 2/3 regress slightly. Do not promote standalone; may only survive as already-OOF input to another cross-fitted layer.

### Pretrained multilingual bi-encoder — insufficient standalone

OOF `0.5318080650341337`, only `+0.0002552941707169021`; raw cosine `~0.3120`. Ready-made item embeddings alone are insufficient; supervised item-space produced the real neural gain.

### First `ruBert-base` pair teacher — INTEGRATION FAIL, not model REJECT

Run `31485127564`, source `00cc48dca806752d92496fa79f703a9ce3bcce63`, failed before comparable OOF. Exact error: `build_reranker_examples() missing 1 required positional argument: 'attribute_importance'`. The runner used a stale helper call. Fix and integration-test the composed path before rerunning; never invent a teacher score.

## Active branch at this snapshot

- field-aware weak ranking teacher: run `31486298300`, source `411a5349fe731506757fdc1a3c8857a370225fb8` — **in progress**.

No metric is claimed until the aggregate completes.

## Debugging / implementation lessons

- >11M weak rows: deterministic PyArrow streaming/bounded sampling only; never full-table pandas before Transformer load.
- Physical MPS batch 96 can OOM; distinguish resource failure from model evidence and preserve effective batch with accumulation when appropriate.
- TF-IDF tests verify unseen transform/symmetry/finite-bounded behavior, not a hand-written OOV ranking.
- Read exact failing test before blaming the latest code; an apparent serializer failure was the next intentionally RED embedding test.
- Weak rows must exclude held-fold and sealed-gold items.
- Heavy workflow helper composition needs integration tests; first ruBERT teacher is the canonical failure example.
- Direct attribute likelihood addition is harmful; explicit attribute features are beneficial. Do not conflate the two approaches.

## Memora operational state and audit findings

- Hardened Memora pin: `bc64ff745a9b2c0e6245e0137654f041fba0c155`; SQLite + TF-IDF local only; LLM/graph/auto-capture disabled.
- Earlier v5 memory runs `31481012401` and `31482891498` failed before ingest because docs-triggering commits landed during intentionally RED TDD. In `31482891498`, collection failed on missing `v5_weighted_specialists`; later GREEN code did not retroactively checkpoint.
- Do not weaken Memora test gates. Checkpoint only from GREEN state.
- A memory audit found `scripts/memory_ingest.py` omitted machine-readable `experiments/CURRENT.json` and `v*/SAFE_METRICS.json`; a regression test now requires both as canonical sources.

## Current action

- production fallback: **v2**, hidden `0.2583231811423486`;
- current honest dev best: **explicit attributes `0.5683065131240066`**;
- target gap: `0.0316934868759934` to `0.60`;
- sealed gold: `80,444`, unopened;
- v5 submit: not retained.

Continue only on the immutable split; do not open gold to guide the next experiment. A v5 submission can be promoted only after candidate/config/preprocessing freeze, one-shot sealed-gold evaluation and organizer runtime/package gates.
