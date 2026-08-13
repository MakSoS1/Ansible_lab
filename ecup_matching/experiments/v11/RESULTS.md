# E-CUP Matching — v11 final keeper

Status: **COMPLETED / SUBMISSION READY**

Date: 2026-08-13

## Selected keeper

`no_teacher + contrastive/structured CPU-GPU overlap + frozen target-free graph`

This is intentionally the same executable/model payload as the fixed v10 no-teacher keeper. v11 does not claim a new model where none exists: the improvement in evidence is a new selection audit plus a fresh exact-archive organizer-like runtime gate with materially stricter limits. The archive is republished under `submissions/v11/final/` so there is one unambiguous submit target.

## Frozen quality evidence

Immutable development protocol: 285,210 development rows, five component-disjoint folds, official 20-category Macro AP, sealed gold unopened.

| Candidate | Strict OOF | Fold-local graph OOF | Target-stress mean | Runtime architecture |
|---|---:|---:|---:|---|
| structured_only | `0.5808404005946962` | `0.5821464487980773` | `0.4355474106077095` | CPU structured only |
| **no_teacher (keeper)** | **`0.5931387077244183`** | **`0.5950413762943735`** | **`0.44961526826354`** | contrastive GPU + structured CPU overlap, no teacher |
| no_contrastive | `0.5928725263319` | `0.5978943607354008` | `0.45353679907723865` | pair teacher + structured, no contrastive |

The locally stronger `no_contrastive` candidate is not the keeper because its pair-scaled teacher is a materially higher platform-time risk. The selected `no_teacher` candidate is the strongest measured quality/runtime tradeoff.

`0.44961526826354` is a target-prevalence stress diagnostic, not a leaderboard score. The owner-reported v7 leaderboard value `~0.36` remains an external anchor and was not used for fitting. Actual v11 leaderboard score is unknown until the platform scores the exact archive.

## Exact submission bytes

User-facing filename: `ecup-v11-faststack-graph-0.5950413763-submission.zip`

Canonical executable payload:

- bytes: **`480,249,754`**;
- SHA-256: **`0d91c8790c9bcaaf3a5e1fb120ed55e179090cad71bac87a796d9294e4ad110a`**;
- teacher assets: `0`;
- contrastive runtime: present;
- structured runtime: present;
- root `run.py`: present;
- root `metadata.json`: present;
- organizer image: `odsai/ecup26-matching-baseline:1.0`;
- entry point: `python -u run.py`.

The v11 filename is a copy/alias of the exact fixed-v10 payload, so the SHA remains identical by design.

## Fresh exact-archive runtime proof

Private GPU run `MakSoS1/gpu-dispatch#31712095181`, job `94487396402`, exact SHA above, NVIDIA GeForce RTX 2060 SUPER, organizer image `odsai/ecup26-matching-baseline:1.0`.

Timer scope: safe ZIP extraction + Docker inference + output validation. Fixture preparation and network transfer are outside the timer.

| Gate | Rows | Extraction | Full wall | v11 acceptance | Result |
|---|---:|---:|---:|---:|---|
| public-size | `115,000` | `2.039450411 s` | **`161.906942010 s`** | `<180 s` | **PASS** |
| private-size | `275,000` | `2.791927307 s` | **`379.244452365 s`** | `<420 s` | **PASS** |

Private-size output contained `271,964` unique finite scores with exact pair order. Internal `run.py` time was `358.08 s`; the outer wall above is authoritative for the gate.

For context, the old compact v9 organizer-like wall on the same RTX class was ~`293.57 s` / `646.95 s` for 115k / 275k. v11 therefore cuts measured wall by roughly 45% / 41%, giving materially more platform margin. This is strong runtime evidence, not a guarantee about the external platform.

## Verification

Public branch verification run `31712646355`, job `94489299438`:

- **450 passed**;
- **6 skipped**;
- `memory_policy.py`: **OK**.

Exact keeper export run `31712297949`, artifact `9185774794`; the inner submission ZIP was re-hashed after export and matches the canonical SHA and byte size.

## Binding rules

- sealed gold opened: **false**;
- gold rows scored: **0**;
- production refit is not validation;
- target-stress is not leaderboard AP;
- owner-reported v7 leaderboard score is not training data;
- actual v11 leaderboard score remains `null` until measured by the competition platform;
- do not submit any archive whose SHA differs from the canonical SHA above without re-running the package/runtime gates.
