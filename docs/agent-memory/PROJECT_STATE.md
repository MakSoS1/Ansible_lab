# E-CUP Matching — Project State

Updated: **2026-08-15** — current iteration **v14 new architecture**

## Executive state

The best **measured external** result remains v12 Public LB `0.3798116204`. v13 B returned `0.3783781653` despite the higher local fold0 score `0.7086611385531062`, proving that near-neighbour local ordering can invert on the Public LB.

The earlier package `v14-v12-category-gated-residual` is now a **superseded technical fallback**, not the current v14. The active work is a genuinely new item-centric architecture with reusable item encodings and compressed pair-conditioned cross-attention. Current candidates use **zero LLM-labelled rows** and never score the sealed gold.

## External anchors

| Candidate | Comparable local diagnostic | Public LB | Meaning |
|---|---:|---:|---|
| v7 | ~`0.70238` | `0.3655833314` | first reliable one-CrossEncoder anchor |
| v12 | `0.7059297810308699` | **`0.3798116204`** | best measured external result |
| v13 B | `0.7086611385531062` | `0.3783781653` | measured negative anchor; local ordering inverted |

No v14 Public LB should be claimed until an exact final archive is submitted.

## Immutable validation / safety state

- Human rows: `365654`.
- Development rows: `285210`.
- Sealed gold: `80444` rows.
- Five component-disjoint development folds.
- Historical split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Canonical row-map SHA-256: `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`.
- Dev↔sealed item overlap: `0`.
- Train↔held item overlap: `0` in every canonical fold.
- `gold_metric_opened=false`, `gold_rows_scored=0`.

The row map is the authoritative historical validation identity.

## LLM weak-label policy

Canonical weak file `matches_llm.parquet` has `11,187,780` rows but exact human-pair overlap `0`. Therefore its class-label precision cannot be established from the existing corpus, and confidence/extreme soft targets are not accepted as proof of correctness.

**Current v14 candidates use zero LLM-labelled rows.** LLM supervision stays denied unless a separate human-controlled audit demonstrates a benefit.

## New architecture evidence

### A0 — plain item-centric LateInteraction

- fold0 Macro AP: `0.5486140975180157`;
- human-only;
- rejected: far below v12 `0.7059297810308699`.

### A1 — A0 + human hard-negative repeats

- fold0 Macro AP: `0.5422162762826607`;
- rejected: worse than A0 by ~`0.00640`.

### A2 — component-closure continuation

- cancelled and rejected without spending another full training cycle after A0/A1 showed sampler-only changes were not credible.

### A3 — LateInteraction + category MoE + ranking

- run `31887218705`;
- evidence artifact `9248080049`;
- fold0 Macro AP: `0.3222800376478955`;
- human-only;
- rejected: category specialization/ranking on top of plain LateInteraction collapsed quality.

## Active architecture

### A5 — ruBERT compressed cross interaction

`item encoder -> reusable token/item cache -> learned 12-slot compression -> bidirectional tiny cross-attention -> symmetric category-conditioned score`

This restores pair-conditioned reasoning without returning to a full concatenated pair CrossEncoder. Human-only fold0 run `31891601603` is active after a read-only `py_compile` cache issue was fixed.

### A8 — Granite-97M compressed cross interaction

Same compressed pair-conditioned architecture, but with pinned multilingual retrieval encoder `ibm-granite/granite-embedding-97m-multilingual-r2`, revision `c61e626a6255c490879d0af885078b61929d51f6`, weights SHA-256 `f3ea88b230492811046145513710e76b4cc8c2ad49e8708da0e7247e548903be`.

A previous A8 attempt reached model construction but ModernBERT automatically invoked TorchInductor/Triton and tried to write a compiler cache under read-only `/root`. Train, production and submission paths now explicitly force `reference_compile=false`; the failed attempts are infrastructure evidence only, not quality evidence.

## Prebuilt escalation if A5/A8 are still below the credible quality region

- **A12** — Granite compressed cross-attention fused with fold-safe typed `features_v2` signals: model code, quantity, brand, same-key attribute agreement/conflict.
- **A6** — fold-safe LLM-free distillation into the compressed-cross student; weak retrieval rows contribute pair IDs only, not the legacy LLM target.
- **A10** — stronger multilingual E5-base encoder under the same compressed-cross architecture, conditional on runtime feasibility.
- setwise candidate-group teacher is available as an offline teacher if pairwise students remain insufficient.

## Promotion and completion gate

A new architecture is not promoted because it is novel. It must first recover to a credible fold0 region. Only then:

1. run exact five-fold OOF over all `285210` development rows;
2. verify exact OOF coverage, zero item leakage, and zero sealed-gold scoring;
3. full-development production refit;
4. build a single-checkpoint offline archive with no raw competition data;
5. run the exact organizer-shaped `<60 s` Check on final ZIP bytes;
6. upload to private Hugging Face and re-download to verify exact byte count/SHA.

## Legacy residual fallback

The earlier `ecup-v14-v12-category-gated-residual-submission.zip` remains reproducible and runtime-verified:

- SHA-256 `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`;
- diagnostic fold0 `0.7065769713851786`;
- organizer-shaped Check `28.810029840000425 s / 60 s` PASS.

It is retained only as a safe technical fallback/reference. It is **not** the requested new architecture and should not be submitted as the current v14 while the new architecture research is active.

## Immediate next action

Finish A5 and the eager-path A8 fold0 screens. If neither is competitive, run A12 typed-attribute fusion and then A6 LLM-free distillation before committing GPU time to strict OOF or packaging. v12 remains the best measured external anchor until a new exact archive receives a platform score.
