# E-CUP v15 — field-aware pair CrossEncoder + offline distillation

Status: `in_progress`
Updated: 2026-08-16

Canonical design: `docs/superpowers/specs/2026-08-16-ecup-v15-field-aware-distillation-design.md`
Implementation plan: `docs/superpowers/plans/2026-08-16-ecup-v15-field-aware-distillation.md`

## Objective

Move beyond the v12/v13 Public-LB plateau (`v12=0.379811620418641`, `v13=0.37837816527590995`) toward a target Public Macro AP of `0.50` without repeating the runtime failures of the structured multi-model branch or the quality loss of the pure item-centric v14 branch.

## Primary hypothesis

Because candidate retrieval is already supplied, the task is primarily fine-grained pair reranking. Full pair-conditioned token interaction remains a strong inductive bias. The next architecture therefore keeps one compact full pair CrossEncoder at inference while improving: field-aware representation, deterministic attribute alignment, lightweight category specialization, macro-oriented training, and offline supervision through a stronger teacher over informative unlabelled retrieval candidates.

## Canonical validation

- human rows: `365654`
- development rows: `285210`
- sealed gold: `80444`, unopened
- 5 component-disjoint folds
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`
- canonical rowmap SHA-256: `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`
- official metric: unweighted mean of per-category `average_precision_score` across all 20 categories

Fold0 is only a screen. A keeper requires strict five-fold OOF covering exactly all `285210` development rows with no duplicate indexes and zero train/held item overlap.

## Architecture ladder

1. `A0 field` — v12-compatible single pair CrossEncoder family; human-only; field-aware deterministic serialization.
2. `A1 attrs` — A0 + normalized structured JSON attributes, model/SKU and conservative numeric/unit normalization.
3. `A2 typed` — A1 + deterministic typed symmetric pair features fused into the neural head; still one Transformer checkpoint.
4. `A3 category` — A2 + lightweight category-conditioned residual head.
5. `A4 macro` — A3 + category-balanced/macro-oriented training control.
6. `B0 teacher` — stronger human-only field-aware teacher; must materially beat the chosen student before pseudo-label generation.
7. `B1 select` — treat historical weak parquet as an unlabelled `id1,id2` candidate graph and select informative pairs without reading legacy target labels.
8. `B2 score` — fold-safe teacher soft-scoring of the selected candidates.
9. `B3 distill` — student distillation followed by clean human recovery.
10. `C optional` — only if needed, separately audited reproducible open-license model relabeling.

## Label policy

- Human labels are authoritative.
- Historical LLM/weak `target` is quarantined by default.
- Historical weak `id1,id2` topology may be used as an unlabelled candidate source.
- New teacher targets must be reproducible and fold-safe.
- No held/sealed item identity leakage into fold training/distillation pools.
- Sealed gold remains unopened.

## Screen / promotion gates

All deltas are against the identical v12-compatible reference on the same held rows.

- `delta < +0.005`: research evidence only by default.
- `+0.005 <= delta < +0.010`: inspect category/hard-slice stability before additional GPU spend.
- `delta >= +0.010`: strong candidate for strict five-fold OOF.

These are GPU-budget gates only, not a local-to-Public calibration.

Strict keeper additionally requires category evidence, no catastrophic category regression, bootstrap-positive delta stability, runtime/package gates, and complete source/model/archive provenance.

## Runtime contract

Final inference:

- one tokenizer;
- one field-aware pair Transformer checkpoint;
- deterministic parsing/typed-feature fusion only;
- no teacher, HGB, TF-IDF, graph engine, second Transformer, network or dynamic download;
- referenced supplied items only, never a full item-universe scan;
- parse/cache normalized fields once per item;
- batched GPU inference with dynamic padding;
- exact final ZIP bytes must pass organizer-shaped checks;
- internal Check target `<=50s` for >=10s headroom under the organizer 60s Check limit.

## Parallel v14 status

Existing v14 runs, including A17, are intentionally left untouched and may finish independently. Their results will be recorded later as parallel legacy/new-architecture evidence. v15 work must not cancel or mutate them.

## Repository contract

`Ansible_lab` is canonical for architecture, source, experiment state and Memora-backed memory. `gpu-dispatch` is the executor and must bind every job to an exact public source SHA, split/rowmap identity, architecture family, role and label policy.

## Expected overnight evidence

Prioritize causal information rather than arbitrary ZIP count: A0/A1/A2 first, A3/A4 if time allows, and B0 teacher preparation/training only after the student path is validated. Production packaging remains gated by quality and runtime; a completed screen is not automatically a submit candidate.
