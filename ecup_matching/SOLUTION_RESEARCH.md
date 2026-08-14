# E-CUP 2026 Matching — current solution research

Updated: 2026-08-15.

This document records the current research conclusion after v1–v13. It contains no raw competition rows or private model weights.

## Task objective

Pairwise product identity ranking after candidate retrieval. Official metric is unweighted Macro Average Precision over 20 categories. Evaluation is offline with strict runtime limits, so model quality and startup/throughput are inseparable selection criteria.

## Data facts

- `365,654` human-labelled pairs over 20 categories.
- Immutable research split: `285,210` development + `80,444` sealed gold, five component-disjoint folds, zero cross-split item overlap, SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Weak pool: `11,187,780` rows.
- Human binary prevalence `0.2567727961`; weak soft-target mean `0.2435606726`.
- Weak candidate structure is materially higher-degree/harder than the human pair table.

## What changed after the early hybrid-cascade hypothesis

Early research favored a structured/bi-encoder/teacher/cascade system. v5 proved that structured, sparse, contrastive and teacher signals all contain useful information: strict local OOF crossed `0.60`. However v9–v11 showed that evaluating these branches directly at submission time is too fragile under the organizer runtime contract.

The resulting architecture rule is now:

```text
OFFLINE TRAINING:
retrieval groups -> structured/semantic/teacher signals -> soft supervision/ranking curriculum -> one student

SUBMISSION:
item1 + item2 -> one ruBERT CrossEncoder -> score
```

Complexity is still valuable, but it belongs mainly in data selection, teacher generation and distillation rather than in `run.py`.

## External evidence

- v7 single CrossEncoder: local fold0 ~`0.70238`, Public LB `0.3655833314`.
- v12 same runtime with stronger weak supervision: fold0 `0.7059297810`, Public LB `0.3798116204`.

This is the strongest observed evidence that retrieval-aligned weak supervision can improve the actual test distribution without changing inference cost. The local metric magnitude is not calibrated to LB.

## Validation v3

The goal is not to force `local AP == LB AP`. It is to rank candidate direction under retrieval-like shift.

Rules:

1. Human binary truth only for validation labels.
2. Component-disjoint held items.
3. Retrieval-hard negative/candidate lists rather than random pair negatives.
4. Category-local prevalence, degree, lexical hardness, text length and attribute-density strata.
5. Multiple deterministic stress replicas; p05 robustness is primary, mean secondary.
6. External leaderboard anchors are used only to test directional usefulness, never as row labels.

## Runtime research

Historical v11 full-item Check forensic: `60.033 s` timeout. This showed that fixed model/data startup and item scanning can kill a submission independently of large-N throughput.

v13 therefore retains one CrossEncoder. Pair-score-only graph post-processing was separately tested because it is cheap (~`1.29 s / 275k`), but rejected because its predeclared variants did not improve v7/v8/v12 consistently.

## v13 retrieval-topology finding

The retained v7/v12 weak curriculum sampled individual weak rows and canonicalized pair orientation, destroying original retrieval candidate groups. It also excluded ambiguous `0.30–0.70` weak targets.

The v13 causal ladder was designed to isolate these factors:

- B/groupweak: preserve complete original retrieval-anchor groups and orientation, change nothing else;
- C/listwise: only then change ranking objective;
- D/all-soft: only then restore ambiguous weak candidates with bounded confidence.

B improved fold0 from v12 `0.7059297810` to `0.7086611386` while preserving identical inference shape. C2/ListNet was rejected in equal-exposure testing. Therefore B was selected as the next external candidate.

## Current candidate

v13 B/groupweak:

- frozen Validation-v3 p05 `0.5690974845`, mean `0.6869505675`;
- single ruBERT CrossEncoder runtime;
- organizer-shaped Check `26.1353473 s / 60 s` PASS;
- exact ZIP SHA `f4b7aad36c8d293a3939d9fb2ce7f91cff1bd8381c870015b2f16ea65a17badb`;
- private-HF roundtrip verified;
- Public LB pending.

## Current research priority

First obtain a real Public-LB score for the exact v13 B archive. Then continue retrieval-aligned training/distillation only when the frozen proxy and new external anchor justify it. Do not reintroduce CPU-heavy inference branches merely to recover local v5 signals.
