# E-CUP Matching — Experiment Index

This file is the short registry. Detailed metrics live in each experiment's `RESULTS.md`.

| Version | Status | Model / idea | Validation | Macro AP | Runtime / execution evidence | Private artifact | Next decision |
|---|---|---|---|---:|---|---|---|
| v1 | completed | structured/lexical HGB | old item-disjoint split, 73,131 pairs | local 0.4961654895; hidden 0.2345852292 | organizer smoke passed | `submissions/v1/ecup-v1-submission.zip` | historical |
| v2 | completed / **production anchor** | product-aware structured + confidence-filtered weak labels | old 73,131-row split | local 0.5010008995; **hidden 0.2583231811** | 275k offline organizer benchmark within limit | `submissions/v2/ecup-v2-submission.zip` | immutable hidden fallback during v5 |
| v3 | completed / historical | v2b + `rubert-tiny2` global blend | old 73,131-row split | local 0.5254642646; hidden canonical 0.2481015189 | exact-image canonical smoke passed | SHA `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660` | historical |
| v4 | completed / historical | v3 + cross-fitted regularized per-category routing | old validation | local OOF 0.5276431099; hidden 0.2531285195 | exact organizer-image freeze passed | SHA `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849` | historical |
| v5 | **in progress** | immutable sealed validation + category/weak/sparse/supervised item-space ladder | **285,210 dev / 80,444 sealed gold / 5 folds / 0 overlap**, SHA `aae58f...eb55b` | **current dev best 0.5662217063 OOF**; gold unopened | contrastive and sparse both improve all five folds | private OOF under `experiments/v5/`; no retained submit | continue honest ladder toward 0.60; v2 remains production fallback |

## v5 development ladder — canonical snapshot

| Step | Status | OOF Macro AP | Delta | Fold consistency | Evidence / consequence |
|---|---|---:|---:|---|---|
| human structured audit baseline | BASE | 0.5315527709 | — | stable 5-fold spread | immutable validation baseline |
| category-specialist HGB | **KEEP** | 0.5476780661 | +0.0161252953 vs audit | all 5 improve | retained structured base |
| direct attribute likelihood score addition | **REJECT** | 0.5232189037 | -0.0083338672 vs audit | all 5 regress | never directly add/tune rescue scalar on same folds |
| pretrained multilingual bi-encoder stack | insufficient standalone | 0.5318080650 | +0.0002552942 vs audit | tiny positive | task supervision required |
| fold-weighted category specialists | diagnostic only | 0.5498696732 | +0.0021916070 vs category | folds 2/3 regress | OOF stack input only |
| leakage-safe weak category specialists | **KEEP** | 0.5514237339 | +0.0037456677 vs category | all 5 improve | weak labels useful under held/gold item exclusion |
| category + weighted + pretrained meta-OOF combo | **KEEP intermediate** | 0.5595125314 | +0.0118344653 vs category | all 5 improve | valid cross-fitted stack |
| strict train-only sparse TF-IDF specialists | **KEEP** | **0.5651306839** | **+0.0174526177 vs category** | **all 5 improve** | rare model/SKU token weighting transfers |
| supervised contrastive item-space stack | **CURRENT DEV BEST / KEEP** | **0.5662217063** | **+0.0185436401 vs category** | **all 5 improve** | task supervision turns item-space into a strong signal |
| first `ruBert-base` pair teacher | **FAIL before metrics** | — | — | all folds fail pre-score | stale helper API: missing `attribute_importance`; not model REJECT |
| explicit per-key attribute specialists | running | — | — | — | run `31485990777`; no metric claimed yet |
| field-aware weak ranking teacher | running | — | — | — | run `31486298300`; no metric claimed yet |

## Current development-best fold AP

Supervised contrastive stack:

- fold 0: `0.5692965046798911`;
- fold 1: `0.5683388560864314`;
- fold 2: `0.5694312406050667`;
- fold 3: `0.5632500994392833`;
- fold 4: `0.5684466083884651`.

Category-base fold AP for comparison:

- `0.5504494811533941`, `0.5478362369858445`, `0.554178032571296`, `0.5453862553120211`, `0.5500737968478944`.

## Important v5 run/artifact evidence

- validation audit: run `31479778679`, source `93e4396330997c41bfb309f449f1dcb79a5e4db6`, private `experiments/v5/validation/93e439633099`;
- category specialists: private `experiments/v5/category/e885961388d1`;
- weighted specialists: run `31483353777`, artifact `9098324849`;
- weak specialists: run `31484641329`, source `319993a469cfa37770d66cfaf1b2203515dc9841`, artifact `9099098118`;
- OOF combo: run `31485240666`, source `7a1c1764a2bdda8f007b9bfea7d088911623e7f0`, artifact `9098856613`;
- supervised contrastive: run `31483288887`, source `b30821f613bf7051da51c42b64c7f79361d5619c`, private `experiments/v5/contrastive-sprint/b30821f613bf/aggregate`, artifact `9099713308`;
- strict sparse: run `31485396599`, source `634ee66890c39ad97c0fa725135b1b00e56ac126`, artifact `9099873750`;
- first ruBERT teacher: run `31485127564`, source `00cc48dca806752d92496fa79f703a9ce3bcce63`, integration failure before predictions.

## Validation facts

- Human rows: `365,654`.
- Components: `345,654`.
- Development rows: `285,210`.
- Sealed gold rows: `80,444`.
- Split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Gold metric opened: **false**; rows scored: **0**.

## Required interpretation

- Production best = v2 by observed hidden leaderboard score; development best = current v5 honest OOF. They are intentionally different.
- `0.60` is judged on the immutable v5 development OOF. Never move the split or repeatedly open gold to reach it.
- Sealed gold is one-shot post-freeze verification, not a tuning set.
- Infrastructure/integration failures are not model scores.
- Cross-fitted stack inputs may include weak standalone signals only when every layer remains truly OOF.
- Private raw data, OOF predictions, learned weights, model artifacts and Memora DB never belong in public Git.
