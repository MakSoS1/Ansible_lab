# E-CUP Matching — Experiment Index

Short registry; detailed evidence is in each experiment `RESULTS.md`.

| Version | Status | Core idea | Validation | Macro AP | Interpretation |
|---|---|---|---|---:|---|
| v1 | completed / historical | structured lexical HGB | old 73,131 item-disjoint holdout | local 0.4961654895; hidden 0.2345852292 | historical |
| v2 | completed / **production anchor** | product-aware structured + weak curriculum | old holdout | local 0.5010008995; **hidden 0.2583231811** | immutable hidden fallback |
| v3 | completed / historical | v2b + `rubert-tiny2` blend | old holdout | local 0.5254642646; hidden 0.2481015189 | historical |
| v4 | completed / historical | cross-fitted regularized category routing | old holdout | local OOF 0.5276431099; hidden 0.2531285195 | historical |
| v5 | **in progress** | immutable sealed CV + category/weak/sparse/neural/explicit-key ladder | **285,210 dev / 80,444 sealed gold / 5 folds / 0 overlap**, SHA `aae58f...eb55b` | **current dev best 0.5683065131 OOF** | continue honest ladder; gold unopened; v2 remains production fallback |

## v5 ladder — canonical snapshot

| Step | Status | OOF Macro AP | Delta | Fold evidence |
|---|---|---:|---:|---|
| human structured audit | BASE | 0.5315527709 | — | stable folds 0.52939–0.53751 |
| category-specialist HGB | **KEEP** | 0.5476780661 | +0.0161252953 vs audit | all 5 improve |
| direct attribute likelihood shift | **REJECT** | 0.5232189037 | -0.0083338672 vs audit | all 5 regress |
| pretrained multilingual bi-encoder | insufficient standalone | 0.5318080650 | +0.0002552942 vs audit | tiny positive only |
| fold-weighted category specialists | diagnostic OOF input only | 0.5498696732 | +0.0021916070 vs category | folds 2/3 regress |
| leakage-safe weak category specialists | **KEEP** | 0.5514237339 | +0.0037456677 vs category | all 5 improve |
| cross-fitted category+weighted+pretrained combo | **KEEP intermediate** | 0.5595125314 | +0.0118344653 vs category | all 5 improve |
| strict train-only sparse TF-IDF specialists | **KEEP** | 0.5651306839 | +0.0174526177 vs category | all 5 improve |
| supervised contrastive item-space stack | **KEEP** | 0.5662217063 | +0.0185436401 vs category | all 5 improve |
| explicit per-key attribute specialists | **CURRENT DEV BEST / KEEP** | **0.5683065131** | **+0.0206284470 vs category** | **all 5 improve** |
| first `ruBert-base` pair teacher | integration FAIL before metrics | — | — | stale helper API; not model REJECT |
| field-aware weak ranking teacher | running | — | — | run `31486298300`; no metric claimed |

## Current dev-best fold AP

Explicit attribute specialists:

- fold 0 `0.5706378464826163`, delta `+0.02018836532922219`;
- fold 1 `0.5682631251392076`, delta `+0.020426888153363132`;
- fold 2 `0.5754313094571646`, delta `+0.021253276885868533`;
- fold 3 `0.5633705139683869`, delta `+0.01798425865636577`;
- fold 4 `0.5731185912680369`, delta `+0.02304479442014251`.

Run `31485990777`, source `cb350b4e7ba6bb4a6d283f91bae4d6ea13235d57`, metrics artifact `9100228112`, artifact digest `6417c94041c3443f03acf85227dceb94e65abea668d1b33bc6dc477f41f5a8fb`.

## Important v5 evidence locations

- validation audit: run `31479778679`, source `93e4396330997c41bfb309f449f1dcb79a5e4db6`, private `experiments/v5/validation/93e439633099`;
- category base: private `experiments/v5/category/e885961388d1`;
- weighted: run `31483353777`, artifact `9098324849`;
- weak: run `31484641329`, source `319993a469cfa37770d66cfaf1b2203515dc9841`, artifact `9099098118`;
- combo: run `31485240666`, source `7a1c1764a2bdda8f007b9bfea7d088911623e7f0`, artifact `9098856613`;
- contrastive: run `31483288887`, source `b30821f613bf7051da51c42b64c7f79361d5619c`, private `experiments/v5/contrastive-sprint/b30821f613bf/aggregate`, artifact `9099713308`;
- sparse: run `31485396599`, source `634ee66890c39ad97c0fa725135b1b00e56ac126`, artifact `9099873750`;
- explicit attributes: run `31485990777`, source `cb350b4e7ba6bb4a6d283f91bae4d6ea13235d57`, artifact `9100228112`;
- first ruBERT teacher: run `31485127564`, integration failure before OOF.

## Required interpretation

- Production best = v2 by observed hidden leaderboard; development best = v5 explicit attributes by honest OOF. Never merge them.
- Split SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b` is immutable for v5 development.
- Sealed gold: `80,444` rows, **unopened**, `0` scored; one-shot post-freeze gate only.
- Target `0.60` must be reached honestly on dev OOF; remaining gap from current best is `0.0316934868759934`.
- Infrastructure/integration failures are not model-quality rejections.
- Private data, models, OOF predictions, learned weights, submission ZIPs and Memora DB stay out of public Git.
