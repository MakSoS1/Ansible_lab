# E-CUP 2026 Matching — current solution research

Updated: 2026-08-15.

This document records the current research conclusion after v1–v13 and the active v14 new-architecture work. It contains no raw competition rows or private model weights.

## Task objective

Pairwise product identity ranking after candidate retrieval. Official metric is unweighted Macro Average Precision over 20 categories. Evaluation is offline with strict runtime limits, so model quality and startup/throughput are inseparable selection criteria.

## Immutable data / validation facts

- `365,654` human-labelled pairs over 20 categories.
- `285,210` development rows + `80,444` sealed gold rows.
- Five component-disjoint development folds with zero train/held item overlap.
- Historical split SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Canonical row-map SHA `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`.
- Sealed gold remains unopened: `gold_metric_opened=false`, `gold_rows_scored=0`.
- Weak retrieval/LLM pool: `11,187,780` rows, but exact human-pair overlap is `0`; its label precision is therefore not established by the existing corpus.

## External evidence and the v13 lesson

| Candidate | Fold0 Macro AP | Public LB |
|---|---:|---:|
| v7 | ~`0.70238` | `0.3655833314` |
| v12 | `0.7059297810308699` | **`0.3798116204`** |
| v13 B/groupweak | `0.7086611385531062` | `0.3783781653` |

v13 improved fold0 but lost `0.0014334551` Public LB to v12. Therefore a near-neighbour local gain is not sufficient evidence for promotion. Local validation remains useful for rejecting broken architectures and comparing large structural changes, but it is not converted into a predicted leaderboard score.

The exact v13 archive was runtime-valid (`26.1353473 s / 60 s` organizer-shaped Check), so its external regression is a quality/distribution problem rather than a crash or packaging failure.

## Why v14 is no longer a CrossEncoder refinement

The small v12 category-gated residual was built and runtime-verified, but it is now retained only as a technical fallback. Its diagnostic gain (`0.7065769713851786` vs v12 `0.7059297810308699`) is too small to justify treating it as the requested architectural step-change.

Active v14 therefore changes the inference graph itself:

```text
item A -> shared encoder -> reusable item/token representation -> compact slots -\
                                                                        -> pair-conditioned tiny cross-attention -> symmetric score
item B -> shared encoder -> reusable item/token representation -> compact slots -/
```

The expensive Transformer encoding stays item-local and can be cached once per unique product. Pair-specific reasoning happens only over a small slot representation rather than over one concatenated 256-token CrossEncoder sequence.

## New-architecture evidence

### A0 — plain item-centric LateInteraction

- human-only;
- component supervised contrastive objective;
- bidirectional token MaxSim + symmetric pooled interaction head;
- fold0 Macro AP **`0.5486140975180157`**;
- **rejected**.

Conclusion: item independence + MaxSim loses too much pair-conditioned reasoning.

### A1 — A0 + human hard-negative repeats

- fold0 **`0.5422162762826607`**;
- **rejected**.

Sampler changes do not repair the architecture.

### A2 — positive component closure

Cancelled after A0/A1 rather than spending another full RTX cycle on a sampler-only change. It has no quality metric and must not be treated as an evaluated winner.

### A3 — LateInteraction + category MoE + category-local ranking

- run `31887218705`, artifact `9248080049`;
- fold0 **`0.3222800376478955`**;
- human-only;
- **rejected**.

Category specialization/ranking on top of plain LateInteraction made the architecture substantially worse.

### A5 — compressed pair-conditioned cross interaction

Active ruBERT candidate:

- independent item encoding;
- 12 learned reusable token slots per item;
- shared bidirectional tiny cross-attention over slots;
- symmetric category-conditioned head;
- category-local ranking + human component SupCon;
- **zero LLM-labelled rows**.

This is the minimum viable new architecture because it restores pair-conditioned reasoning without returning to full pair concatenation.

### A8 — retrieval-pretrained compact encoder + compressed cross interaction

Same pair-conditioned architecture with pinned `ibm-granite/granite-embedding-97m-multilingual-r2`, revision `c61e626a6255c490879d0af885078b61929d51f6`, weights SHA `f3ea88b230492811046145513710e76b4cc8c2ad49e8708da0e7247e548903be`.

ModernBERT implicit compiler use was explicitly disabled for training/production/runtime because it added an unnecessary TorchInductor/Triton startup/cache path in the read-only container. Failed pre-fix attempts are infrastructure evidence only, not quality evidence.

### A12 — typed structured fusion

Prepared reserve if A5/A8 remain below the credible quality region:

- Granite compressed-cross backbone;
- fold-train-only attribute importance;
- typed `features_v2` signals inside the neural head: model code, quantities, brand, attribute-key/value agreement/conflict;
- human-only labels;
- eager ModernBERT path.

### A6 — LLM-free distillation

Prepared reserve using the large retrieval graph without trusting historical LLM labels:

- Arrow reads only `id1,id2` from the weak graph; legacy `target` is never requested;
- complete human item universe is excluded from candidate distillation rows;
- soft targets come from fold-safe human-trained neural + structured teachers;
- final submission model remains the compressed-cross student.

## Current LLM-label decision

The user hypothesis that noisy LLM labels may ruin quality is treated as an explicit gate. Existing weak labels are **not admitted** because the current corpus does not provide controlled human overlap to measure their precision. v14 A0/A1/A3/A5/A8/A12 use zero LLM-labelled rows. A6 reuses only the retrieval graph topology and teacher-generates new targets from human-trained models.

LLM labels may return only after a separate human-stratified audit proves reliability on hard cases (model-number, capacity, size, edition/year/color, accessory-vs-main-product, high-similarity negatives). Self-reported LLM confidence is not sufficient.

## Promotion protocol

Do not package a candidate merely because it is new. First require credible fold0 recovery toward the v12 quality region. A promoted architecture must then pass, in order:

1. exact five-fold component-disjoint OOF over all `285210` development rows;
2. exact OOF coverage once, zero duplicates, zero train/held item overlap;
3. zero sealed-gold scoring;
4. full-development production refit;
5. one-checkpoint offline archive with no raw competition data and no network;
6. organizer-shaped 1,000-pair supplied-item Check `<60 s` on the exact final ZIP bytes;
7. private Hugging Face upload and byte-for-byte SHA roundtrip.

## Runtime direction

The old v12/v13 one-CrossEncoder runtime remains only an external reference/fallback. The intended new runtime is:

`unique-item encoding once -> CPU/GPU item-slot cache -> cheap pair cross-attention -> predict`.

This preserves pair-conditioned reasoning while allowing repeated products to reuse the expensive encoder output.

## Current research priority

Finish A5 and eager-path A8 fold0. If neither recovers enough quality, run A12 typed-feature fusion and A6 LLM-free distillation before strict OOF. A stronger multilingual E5-base encoder and a setwise candidate-group teacher remain secondary escalation paths. Do not return to weak-sampler tuning of the old CrossEncoder line as the primary v14 strategy.
