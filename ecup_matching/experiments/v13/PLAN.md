# E-CUP Matching — v13 plan

Date: 2026-08-14/15
Status: **next Public-LB candidate packaged and verified; platform score pending**

## Objective

Move from v12 Public LB `0.3798116204` toward first place / `>=0.50` without returning to inference architectures that time out. Keep the fast one-CrossEncoder runtime and move complexity to retrieval-aligned training/distillation.

## Binding constraints

- immutable human split SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- sealed gold unopened during research;
- one ruBERT CrossEncoder / one tokenizer / one checkpoint by default at inference;
- Public-LB results are external experiment anchors, never row labels;
- private HF only for model/submission binaries;
- every final artifact is bound by exact bytes and SHA-256;
- organizer-shaped 1,000-row Check is mandatory before publication.

## Research tracks

### A — external anchors

Record v7 (`0.3655833314`) and v12 (`0.3798116204`) exactly; add spread anchors rather than fitting a calibration from two nearby scores.

### B — measure distribution shift

Measure human and weak prevalence, degree, candidate-list structure and hardness by category. Result: prevalence alone is insufficient; weak retrieval structure is materially higher-degree/harder.

### C — runtime forensics

Re-run historical v11 against a Check-sized full-item stress fixture. Result: `60.033 s` timeout proves fixed/full-item startup can be fatal.

### D — graph post-processing

Reopen pair-score-only graph transforms independently of structured runtime. Predeclare variants, test on v7/v8/v12, retain only if stable. Result: runtime cheap, quality unstable, REJECT.

### E — retrieval-aligned student training

Use a causal ladder:

1. **B / groupweak:** preserve original `_retrieval_anchor` and complete weak candidate groups; do not change model/runtime/loss.
2. **C / listwise:** only after B, alter group ranking objective.
3. **D / all-soft:** only after C, restore ambiguous weak `0.30–0.70` candidates with bounded weights.
4. Distillation/tail repair only after a frozen proxy supports them.

## Selection principle

A new idea must beat its parent on predeclared validation, preserve category/tail safety, and keep the one-CrossEncoder runtime. More complex does not mean better. A new Public-LB candidate may be packaged for external calibration without being called the strict final keeper, but that distinction must be explicit.

## Runtime gates

- binding Check: 1,000 rows, official limit 60 s; target substantial headroom;
- larger public/private stress profiles remain diagnostic for throughput;
- full 4.1-GB item-universe scans are a deliberately conservative diagnostic and must not be conflated with the supplied-item subset contract.

## Current selected external candidate

B / groupweak:

- probe run `31791177120`;
- fold0 `0.7086611385531062`;
- frozen Validation-v3 p05 `0.5690974845`, mean `0.6869505675`;
- production run `31828844182`;
- package run `31829720888`;
- HF roundtrip run `31843423348`;
- Public LB pending.

## Next gate

Submit the exact verified B archive to ODS and record Public LB. Use that result to decide the next research iteration; do not rewrite history based on the result.
