# E-CUP Matching — v10 faststack plan

Status: **COMPLETED / KEEPER**

Date: 2026-08-13

## Objective

Produce a submission that is materially more transferable than the owner-reported v7 leaderboard result (`~0.36`), moves toward the requested `~0.5` region, and is safely below competition runtime limits.

The leaderboard target is an objective, not a claimed score. Actual v10 leaderboard remains unknown until the platform scores the exact keeper.

## Immutable validation contract

- human rows: `365654`;
- development rows: `285210`;
- sealed gold rows: `80444`;
- five component-disjoint outer folds;
- cross-split item overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- official local metric: unweighted Macro AP over exactly 20 categories;
- sealed gold opened: `false`;
- sealed gold rows scored: `0`.

## Candidate search

The tiny-student path (`cointegrated/rubert-tiny2`) was rejected at strict OOF `0.39535352249445677`.

The strict fast-stack comparison under the same frozen graph and 100-trial target-stress protocol was:

| Candidate | Strict OOF | Graph OOF | Graph delta | Positive graph folds | Stress mean |
|---|---:|---:|---:|---:|---:|
| structured_only | `0.5808404005946962` | `0.5821464487980773` | `+0.0013060482033811` | `5/5` | `0.4355474106077095` |
| **no_teacher** | **`0.5931387077244183`** | **`0.5950413762943735`** | **`+0.0019026685699552`** | **`5/5`** | **`0.44961526826354`** |
| no_contrastive | `0.5928725263319` | `0.5978943607354008` | `+0.0050218344035008` | `5/5` | `0.45353679907723865` |

`no_contrastive` was not selected despite the strongest local diagnostics because it retains the pair cross-encoder teacher, the stage that scales with candidate-pair count and conflicts with the runtime objective.

## Selected algorithm

**`no_teacher + frozen target-free graph`**.

Frozen graph:

```text
reciprocal_best_bonus = 0
reciprocal_top3_bonus = 0
endpoint_rank_weight = 0.02
ambiguity_penalty = 0.01
```

The missing teacher signal is the frozen v6 no-teacher surrogate: unweighted mean percentile rank of `weak`, `sparse`, `explicit`, `contrastive`, and `typed_explicit`.

Quality evidence:

- strict raw OOF Macro AP `0.5931387077244183`;
- strict graph OOF Macro AP `0.5950413762943735`;
- graph positive on `5/5` folds;
- target-stress mean `0.44961526826354`;
- target-stress std `0.0016793717`;
- p05/p50/p95 `0.44725891257 / 0.44972458938 / 0.45217563833`.

Target-stress is diagnostic only and is not relabeled as leaderboard performance.

## Runtime architecture

The pair teacher is absent from code path and archive. Runtime overlaps independent work:

1. load input/frozen models;
2. fork structured CPU scoring before CUDA initialization;
3. build only the contrastive legacy text cache;
4. run contrastive GPU embeddings while structured CPU scoring continues;
5. join branches;
6. reproduce no-teacher six-signal composition;
7. run frozen production rank fusion;
8. apply frozen graph;
9. write `id1,id2,predict`.

Larger structured chunks (`20k/25k`) were benchmarked and rejected as slower. A corrected runtime sweep showed batch/worker variants preserved predictions within `rtol=1e-6, atol=1e-7`; no post-gate tuning was allowed to mutate keeper bytes.

## Exact immutable keeper

Build run `31689478925`.

- release tag `ecup-v10-faststack-9de2bc83f878`;
- archive `ecup-v10-no-teacher-graph-0.5950413763-submission.zip`;
- bytes `480249520`;
- SHA-256 `6cebc276f45fc52247db054eb83d2a8110b25d4407cc34b0d5b148a4773c321d`;
- source SHA `9de2bc83f878c87703c3290670f042bfdbb70dfc`;
- teacher checkpoint packaged `false`;
- CPU/GPU overlap `true`;
- contrastive-only text cache `true`.

## Final runtime acceptance

The earlier `<120 s / <250 s` thresholds were exploratory over-strict tuning targets, not organizer rules. They are not the production keeper acceptance criteria.

Production engineering acceptance retained from the established competition runtime protocol is `<330 s` public-size and `<700 s` private-size, deliberately below the nominal project-recorded `360/780 s` organizer budgets.

Exact same SHA on RTX 2060 SUPER with `odsai/ecup26-matching-baseline:1.0`:

| Gate | Rows | Outer inference wall | Internal acceptance | Headroom | Result |
|---|---:|---:|---:|---:|---|
| public-size | `115000` | `173.842174445 s` | `330 s` | `156.157825555 s` | **PASS** |
| private-size | `275000` | `391.608035937 s` | `700 s` | `308.391964063 s` | **PASS** |

Private keeper run `31692817075`, artifact `9178292328`, also validated exact pair order, finite/nonconstant scores and `271964` unique scores. Return code was `0`; pair-teacher checkpoint was absent.

## Publication

HF publication run `31693414226`: **SUCCESS**.

Private paths:

- `submissions/v10/final/ecup-v10-no-teacher-graph-0.5950413763-submission.zip`;
- `submissions/v10/final/V10_KEEPER.json`.

The workflow reverified immutable release bytes/SHA and relisted both HF paths after upload.

## Next external measurement

Submit this exact archive to the competition platform. If the platform returns a score, record it as a new external evidence axis; never rewrite strict OOF or target-stress history to match leaderboard behavior.
