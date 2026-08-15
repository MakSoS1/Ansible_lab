# E-CUP Matching — Project State

Updated: **2026-08-15** — current iteration **v14**

## Executive state

The best **measured external** result remains v12 Public LB `0.3798116204`. v13 B returned `0.3783781653` despite a higher local fold0 score, proving a real local/external ordering inversion. The current submission-ready candidate is **v14 `v14-v12-category-gated-residual`**: the exact v12 one-CrossEncoder parent plus a small human-only category-gated lexical residual that passed item-disjoint cross-fit and an organizer-shaped runtime Check on the exact final ZIP.

v14 Public LB is not measured yet. Do not claim `>0.5` or even `>v12` externally until the platform returns a score.

## External leaderboard anchors

| Candidate | Comparable local diagnostic | Public LB | Meaning |
|---|---:|---:|---|
| v7 | ~`0.70238` | `0.3655833314` | first reliable one-CrossEncoder anchor |
| v12 | `0.7059297810308699` | **`0.3798116204`** | best measured external parent |
| v13 B | `0.7086611385531062` | `0.3783781653` | measured negative anchor; local ordering inverted |
| v14 | `0.7065769713851786` | pending | current runtime-verified submission candidate |

## Immutable validation / safety state

- Human rows: `365654`.
- Development rows: `285210`.
- Sealed gold: `80444` rows.
- Five component-disjoint development folds.
- Historical split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Recovered canonical row-map SHA-256: `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`.
- Row-map source: artifact `9175469673`, run `31680767570`.
- Current data alignment: dev↔sealed item overlap `0`; train↔held item overlap `0` in every fold.
- Sealed gold remains unopened; `gold_metric_opened=false`, `gold_rows_scored=0`.

The row map, not a newly recomputed split in a changed environment, is now the authoritative historical validation identity.

## Why v13 changed the selection policy

v13 B improved the local diagnostics:

- v12 fold0 `0.7059297810308699`;
- v13 B fold0 `0.7086611385531062`.

But Public LB moved the opposite direction:

- v12 `0.3798116204`;
- v13 B `0.3783781653`;
- v13 minus v12 = `-0.0014334551`.

Therefore fold0/Validation-v3 can be useful diagnostics but cannot independently promote a near-neighbour candidate. v13 is retained as a measured negative external anchor.

## Weak-label audit

Canonical weak file `matches_llm.parquet` has `11,187,780` rows. Exact pair overlap with the human truth set is `0`, so positive/negative precision cannot be audited against controlled human labels. Confidence or extreme soft targets are not accepted as substitutes for ground truth.

**v14 uses zero LLM-labelled rows.** The residual is trained from human development labels only.

## v14 architecture research

An item-centric/late-interaction A2 path was implemented with one ruBERT encoder, pooled field vectors, token MaxSim, human hard negatives and component closure. Two engineering issues were found and fixed:

1. component closure failed for endpoints appearing only as `id2`;
2. current split recomputation did not reproduce the historical split SHA, leading to the explicit canonical row-map recovery.

A complete new-Transformer strict five-fold was not finished after projected multi-hour RTX cost. This is recorded as unfinished architecture research, **not** a quality rejection.

## Fast residual ladder

### v1 scalar residual — rejected

- v12 fold0 reproduced exactly: `0.7059297810308699`;
- diagnostic fold0 `0.7060368061079549`;
- delta `+0.00010702507708482134`;
- cross-fit mean `+0.0002866901011532867`;
- only 11/20 categories non-negative.

### v2 category gate — evaluator bug found

The first v2 category logic computed per-category admission correctly but the aggregate cross-fit score still applied residual alpha to categories later rejected by the gate. That aggregate evaluator was inconsistent with the intended production decision rule.

The evaluator was corrected so rejected categories remain exactly at the v12 base score. **Promotion thresholds were not changed.**

### v2 corrected — accepted

Corrected probe run `31882322590`, evidence artifact `9246360741`:

- v12 fold0 `0.7059297810308699`;
- v14 fold0 `0.7065769713851786`;
- delta `+0.0006471903543086022`;
- side0 opposite-half delta `+0.000437006267165585`;
- side1 opposite-half delta `+0.000734831086673049`;
- cross-fit mean `+0.000585918676919317`;
- admitted categories `6`;
- full-fold categories non-negative vs v12: `20/20`;
- frozen promotion gate: PASS.

Active residual categories and alphas:

- `Автотовары`: `0.025`;
- `Аптека`: `0.025`;
- `Бытовая техника`: `0.10`;
- `Мебель`: `0.05`;
- `Музыкальные инструменты`: `0.075`;
- `Спорт и отдых`: `0.025`.

All other categories preserve the v12 base score exactly.

## v14 final artifact

Packaging/runtime run: `31882572941`.

- filename: `ecup-v14-v12-category-gated-residual-submission.zip`;
- bytes: `663770301`;
- SHA-256: `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`;
- model SHA-256: `b137761de29dd17b5ac058bc51a4cd5d113f3531a1d60071a91a3ae058ac55e6`;
- exact parent v12 archive SHA-256: `a189eb9eaf97ad74c323ef446759c4b42e392f09df8d65327f938b582d01dac1`;
- one safetensors checkpoint;
- no competition parquet/csv in the archive;
- sealed gold unopened;
- zero v14 LLM rows.

### Binding organizer-shaped Check on exact final bytes

- organizer image: `odsai/ecup26-matching-baseline:1.0`;
- pairs: `1000`;
- supplied items: `1999`;
- ZIP extraction: `3.4397788900005253 s`;
- total wall: **`28.810029840000425 s / 60 s`**;
- return code `0`;
- timed out `false`;
- output valid `true`;
- unique scores `910`;
- result: **PASS**.

## Runtime architecture

The retained inference architecture is the measured v12 one-ruBERT CrossEncoder plus a lightweight category-gated six-feature residual. Heavy multi-model/TF-IDF/graph inference remains closed. The exact final archive, not a development source tree, defines the accepted runtime behavior.

## Immediate next action

Submit exactly `ecup-v14-v12-category-gated-residual-submission.zip` with SHA-256 `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e` to ODS and record the measured Public LB in the leaderboard registry and v14 results. Until that happens, v12 `0.3798116204` remains the best measured external anchor.
