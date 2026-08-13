# E-CUP Matching — v10 faststack plan

Status: **RUNTIME GATE IN PROGRESS**

Date: 2026-08-13

## Why v10 exists

The v9 multi-stage architecture and its compact variant both failed on the competition platform with `Container did not finish in time`. Local RTX timing is retained as engineering evidence, but is no longer treated as sufficient evidence that a package will finish on the platform.

v10 therefore changes the inference architecture and makes runtime a hard pre-release gate.

## Immutable validation contract

- development rows: `285,210`;
- sealed-gold rows: `80,444`;
- five component-disjoint outer folds;
- cross-split item overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- metric: unweighted Macro AP over exactly 20 official categories;
- sealed gold stays unopened;
- actual v10 leaderboard score stays unknown until the platform scores the exact archive.

## Rejected tiny-student branch

The initial `cointegrated/rubert-tiny2` student proved that a transformer can be made very fast, but it failed the quality gate.

Full immutable five-fold strict OOF:

`0.39535352249445677`

Fold AP:

- `0.39768678520312284`;
- `0.39292850716572086`;
- `0.4002174014891855`;
- `0.4000653241124591`;
- `0.4018525544121479`.

This candidate is rejected. Its runtime evidence must not be confused with keeper quality evidence.

## Fast-stack comparison

v10 next reused the already strict outer-cross-fitted v6 fast-ablation vectors and evaluated them under one frozen target-free graph rescore.

Frozen graph configuration:

```text
reciprocal_best_bonus = 0
reciprocal_top3_bonus = 0
endpoint_rank_weight = 0.02
ambiguity_penalty = 0.01
```

The same v9 validation-v2 target-stress protocol was used with positive prevalence multiplier `0.566880890615799` and 100 deterministic trials.

| Candidate | Strict OOF | Graph OOF | Graph delta | Positive graph folds | Graph stress mean |
|---|---:|---:|---:|---:|---:|
| structured_only | 0.5808404005946962 | 0.5821464487980773 | +0.0013060482033811 | 5/5 | 0.4355474106077095 |
| **no_teacher** | **0.5931387077244183** | **0.5950413762943735** | **+0.0019026685699552** | **5/5** | **0.44961526826354** |
| no_contrastive | 0.5928725263319 | 0.5978943607354008 | +0.0050218344035008 | 5/5 | 0.45353679907723865 |

`no_contrastive` wins the local diagnostics but retains the pair cross-encoder teacher. That stage scales with candidate-pair count and conflicts with the reason v10 exists. It is rejected as a production direction despite the local metric advantage.

## Selected candidate

Selected candidate:

**`no_teacher + frozen graph`**

Frozen quality evidence:

- raw strict OOF Macro AP: `0.5931387077244183`;
- graph strict OOF Macro AP: `0.5950413762943735`;
- graph improves all 5 immutable folds;
- graph target-stress mean: `0.44961526826354`;
- graph target-stress std: `0.0016793717`;
- graph target-stress p05: `0.44725891257`;
- graph target-stress p50: `0.44972458938`;
- graph target-stress p95: `0.45217563833`;
- pair-teacher checkpoint at inference: **absent**.

The missing teacher signal is reproduced exactly as in the frozen v6 `no_teacher` ablation: the unweighted mean of target-free percentile ranks of `weak`, `sparse`, `explicit`, `contrastive`, and `typed_explicit`.

## Runtime redesign

Historical optimized v6 275k phase timing on RTX 2060 SUPER was approximately:

- structured: `220 s`;
- dual text cache: `34 s`;
- contrastive: `194.5 s`;
- pair teacher at 70% coverage: `239.5 s`;
- meta: `<1 s`.

Removing the teacher alone would still leave the remaining expensive phases sequential. v10 therefore changes scheduling without changing score functions:

1. load inputs and frozen models;
2. before CUDA initialization, fork the CPU structured scorer;
3. parent builds only the exact legacy 700-character contrastive text view; the unused teacher text view is not generated;
4. parent runs contrastive embedding/scoring on GPU while structured CPU work continues;
5. join the structured process;
6. reproduce the frozen no-teacher six-signal composition;
7. apply production category-shrunk/HGB rank fusion;
8. apply frozen target-free graph rescore;
9. write `id1,id2,predict`.

This turns CPU structured scoring and GPU contrastive work into parallel branches instead of adding their wall times.

## Immutable candidate archive

Build run: `31689478925`

Release tag:

`ecup-v10-faststack-9de2bc83f878`

Archive:

`ecup-v10-no-teacher-graph-0.5950413763-submission.zip`

- bytes: `480249520`;
- SHA-256: `6cebc276f45fc52247db054eb83d2a8110b25d4407cc34b0d5b148a4773c321d`;
- source SHA: `9de2bc83f878c87703c3290670f042bfdbb70dfc`;
- teacher checkpoint packaged: `false`;
- CPU/GPU overlap: `true`;
- contrastive-only text cache: `true`.

The archive is an immutable candidate, not yet a keeper, until the exact runtime gate completes.

## Hard runtime gate

Exact runtime workflow run:

`31689794784`

The gate downloads the exact release bytes and verifies SHA-256 and size before extraction.

Internal acceptance thresholds:

- 115k organizer-image outer wall: `<120 s`;
- 275k organizer-image outer wall: `<250 s`.

Additional conditions:

- organizer image `odsai/ecup26-matching-baseline:1.0`;
- CUDA enabled;
- network disabled;
- submission mounted read-only;
- exact output columns `id1,id2,predict`;
- exact row count and pair order;
- finite and nonconstant scores;
- no artificial `[0,1]` constraint after graph rescoring.

These thresholds are deliberately stricter internal engineering gates, not claimed organizer-published limits.

## Leaderboard objective

The objective is to materially exceed the observed v7 leaderboard result and move toward `0.5`, while completing within the external platform time limit. The local OOF and target-stress numbers above are selection evidence only and must not be relabeled as leaderboard measurements.
