# E-CUP Matching — Experiment Index

This file is the short registry. Detailed metrics live in each experiment's `RESULTS.md`.

| Version | Status | Model / idea | Validation | Macro AP | Runtime / execution evidence | Private artifact | Next decision |
|---|---|---|---|---:|---|---|---|
| v1 | completed | structured/lexical HGB | old item-disjoint connected-component split, 73,131 pairs | local 0.4961654895; hidden 0.2345852292 | organizer smoke passed | `submissions/v1/ecup-v1-submission.zip` | historical |
| v2 | completed / **production anchor** | product-aware structured features + confidence-filtered LLM weak labels (`v2b-weak-curriculum`) | old 73,131-row item-disjoint validation | local 0.5010008995; **hidden 0.2583231811** | 275k-pair offline organizer benchmark within limit | `submissions/v2/ecup-v2-submission.zip` | keep immutable as hidden/production fallback during v5 |
| v3 | completed / historical | v2b + `rubert-tiny2` global blend | old 73,131-row validation | local 0.5254642646; hidden canonical 0.2481015189 | exact-image canonical smoke passed | SHA `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660` | immutable historical artifact |
| v4 | completed / historical | v3 scores + component-cross-fitted regularized per-category neural alphas | old 73,131 rows; routing crossfit on 53,131 validation components | local OOF 0.5276431099; hidden 0.2531285195 | exact organizer-image smoke/freeze passed | SHA `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849` | immutable historical artifact; not current production anchor |
| v5 | **in progress** | new sealed validation + category specialists + leakage-safe weak/semantic/stacking ladder | **285,210 dev / 80,444 sealed gold / 5 folds / 0 overlap**, split SHA `aae58f...eb55b` | **current dev best 0.5595125314 OOF**; gold unopened | combo improves all 5 folds; several neural/sparse/explicit branches still active | private OOF under `experiments/v5/`; no retained submit yet | continue honest ladder toward 0.60; freeze before one-shot gold; v2 remains production fallback |

## v5 development ladder — canonical snapshot

| Step | Status | OOF Macro AP | Delta | Fold consistency | Evidence / consequence |
|---|---|---:|---:|---|---|
| v5a human structured audit baseline | BASE | 0.5315527709 | — | folds 0.52939–0.53751 | new immutable split is stable; gold unopened |
| category-specialist HGB | **KEEP** | 0.5476780661 | +0.0161252953 vs audit | all 5 improve | retained structured base |
| direct attribute likelihood score addition | **REJECT** | 0.5232189037 | -0.0083338672 vs audit | all 5 regress | never directly add this evidence or tune a rescue scalar on same folds |
| pretrained multilingual bi-encoder stack | insufficient standalone | 0.5318080650 | +0.0002552942 vs audit | small positive | pretrained item space alone is not enough |
| fold-weighted category specialists | diagnostic only | 0.5498696732 | +0.0021916070 vs category base | folds 2/3 slightly regress | not retained standalone; allowed only as already-OOF stack input |
| leakage-safe weak category specialists | **KEEP** | 0.5514237339 | +0.0037456677 vs category base | **all 5 improve** | weak labels useful when held/gold items excluded |
| category + weighted + pretrained cross-fitted combo | **CURRENT DEV BEST / KEEP** | **0.5595125314** | **+0.0118344653 vs category base** | **all 5 improve by ~+0.012** | current v5 development benchmark; not yet a submission |
| supervised contrastive outer-CV | running | — | — | — | run `31483288887`; initial batch-96 MPS OOM fixed by 24×4 accumulation |
| `ruBert-base` pair teacher outer-CV | running | — | — | — | run `31485127564`; no metric until aggregate completes |
| strict sparse TF-IDF specialists | running | — | — | — | run `31485396599`; vocab/IDF train only on outer-train items |
| explicit per-key attribute specialists | running | — | — | — | run `31485990777`; held labels not used to choose keys |
| field-aware weak ranking teacher | queued at snapshot | — | — | — | run `31486298300`; no metric claimed |

## v5 validation facts

- Human rows: `365,654`.
- Connected item components: `345,654`.
- Development rows: `285,210`.
- Sealed gold rows: **`80,444`**.
- Split SHA-256: **`aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`**.
- Gold metric opened: **false**; gold rows scored: **0**.
- Baseline held-fold AP: `0.5351372013`, `0.5293875047`, `0.5375146979`, `0.5315124247`, `0.5337619678`.
- Current combo held-fold AP: `0.5625800658`, `0.5605549740`, `0.5667063355`, `0.5579194709`, `0.5631351691`.

## v5 important run/artifact evidence

- validation audit: run `31479778679`, source `93e4396330997c41bfb309f449f1dcb79a5e4db6`, private `experiments/v5/validation/93e439633099`;
- category specialists: private `experiments/v5/category/e885961388d1`;
- weighted specialists: run `31483353777`, aggregate private `experiments/v5/weighted-specialists/9df24f7ee133/aggregate`, artifact `9098324849`;
- weak specialists: run `31484641329`, source `319993a469cfa37770d66cfaf1b2203515dc9841`, private `experiments/v5/weak-specialists/319993a469cf/aggregate`, artifact `9099098118`;
- OOF combo: run `31485240666`, source `7a1c1764a2bdda8f007b9bfea7d088911623e7f0`, private `experiments/v5/combo/7a1c1764a2bd`, artifact `9098856613`.

## Historical v2/v3/v4 interpretation

- v2a human + product-aware features: `0.5006971263` old-local Macro AP.
- v2b + 300k confidence-filtered weak labels: `0.5010008995` old-local; now production anchor because hidden score is strongest among submitted v1-v4 candidates.
- v2c naive static hard-negative reweighting: `0.4957263069` — rejected.
- v3 local neural gain and v4 local routing gain are real measurements on the old validation, but hidden scores show they did not transfer monotonically. Do not use them as the main v5 selection gate.
- v4's original `ai-forever/ruBert-base` branch never produced retained comparable evidence; its first RTX attempt failed before metrics. Do not fabricate a score from infrastructure failures.

## Required interpretation

- “Production best” currently means v2 by observed hidden leaderboard score; “development best” currently means v5 combo by sealed-protocol OOF. These are intentionally different.
- The `0.60` stretch target is judged on the immutable v5 development OOF, not by changing the split or repeatedly opening gold.
- Sealed gold is a one-shot post-freeze gate, not a tuning set.
- Threshold accuracy/F1 does not replace category Macro AP.
- A branch may be used as a cross-fitted stack input even if it is not retained standalone, but only when every layer is truly OOF and the final stack proves held-fold gain.
- Private artifacts, raw data, OOF predictions, learned weights and Memora DB never belong in public Git.
