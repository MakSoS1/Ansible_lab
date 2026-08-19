# E-CUP Matching v20 — Audited Data-Centric Rationale Distillation

Status: implementation in progress
Date: 2026-08-19

Canonical design: `docs/superpowers/specs/2026-08-19-ecup-v20-audited-rationale-distillation-design.md`.
Implementation plan: `docs/superpowers/plans/2026-08-19-ecup-v20-data-centric-implementation.md`.

## Why v20 exists

The external evidence says small local architectural gains are not enough. Public LB anchors are v7 `0.3655833314`, v12 `0.379811620418641`, v13B `0.37837816527590995`, v14 `0.38032704703111925`. Human fold-0 misranks v12/v13B/v14, while the only large external movement came with increased weak supervision. The weak corpus covers 12,384,610 products and 11,187,780 pairs, yet historical v12+ training saw only about 210k weak examples per run.

v20 therefore improves supervision density/quality first and keeps the proven one-checkpoint RuBERT CrossEncoder runtime.

## Frozen D0-D10 ladder

- D0: ledger repair and policy hash.
- D1: complete semantic census of weak supervision and real items.
- D2: proxy calibration using frozen v7/v12/v13B/v14 anchors.
- D3: two-teacher audit against fold-safe authoritative human calibration rows.
- D4: target-free real-item candidate generation, two-teacher labelling, statistical admission, gold corpus build.
- D5: data-only RuBERT baseline.
- D6: same data plus rationale multi-task heads.
- D7: keeper plus source-aware mixed replay.
- D8: scale only retained mechanisms.
- D9: frozen two-fold confirmation.
- D10: full-development production refit and exact organizer package.

## Binding rules

- frozen split SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- sealed gold unopened/unscored;
- two independent teachers required for any new LLM-generated label;
- LLM self-confidence never admits a row;
- `UNCERTAIN`/disagreement/checker conflict never receives training weight;
- final runtime is one `ai-forever/ruBert-base` pair CrossEncoder, max length 256, one `.safetensors`;
- no graph/listwise/Granite/residual mechanism is reintroduced without a new isolated ablation;
- hard-negative-only curricula are forbidden by design because v2/v3 rejected them;
- v19 refresh is conditional on explicit v19 pass evidence only.
