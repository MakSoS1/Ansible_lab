# E-CUP Matching — Experiment Index

This is the canonical compact ledger. Every retained experiment must have a row here and a source-backed `experiments/vN/RESULTS.md`.

| Iteration | Status | Model / hypothesis | Validation | Macro AP | Runtime note | Private artifacts | Git source |
|---|---|---|---|---:|---|---|---|
| v1 | completed | `v1-structured-hgb`; deterministic lexical/attribute/numeric features + sklearn HGB, human labels only | item-disjoint 292,523 train / 73,131 valid; overlap 0 | **0.4961654895** | train/eval 308.57 s; 1k-pair offline smoke 1.78 s | `submissions/v1/` in private HF | `ecup_matching/experiments/v1/RESULTS.md` |
| v2 | planned | filtered/confidence-weighted 11M LLM weak labels + hard-negative mining on the fast structured anchor | reuse fixed item-disjoint human validation | — | target: preserve large organizer runtime headroom | `submissions/v2/` when complete | `ecup_matching/experiments/v2/` |

## Rules

- Never overwrite a prior experiment's evidence. Create a new iteration when training/data/validation changes materially.
- All category AP values belong in the iteration `RESULTS.md`, even if only Macro AP is shown here.
- `completed` requires results documentation, policy validation, private artifact verification, and a Memora checkpoint.
- A worse experiment is still recorded if it taught something material; status can be `rejected` with the reason in RESULTS.
