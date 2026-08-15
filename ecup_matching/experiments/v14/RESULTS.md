# E-CUP v14 — New Architecture Results

Status: **in progress — fold0 architecture screening**

The earlier `v14-v12-category-gated-residual` package is retained only as a technical fallback. It is **not** the requested new architecture and must not be treated as the current v14 candidate.

## External anchors

| Version | Comparable fold0 Macro AP | Public LB |
|---|---:|---:|
| v12 | `0.7059297810308699` | **`0.3798116204`** |
| v13 B | `0.7086611385531062` | `0.3783781653` |

v13 improved local fold0 while losing `0.0014334551` Public LB to v12. Therefore fold0 is an architecture screen, not a calibrated leaderboard predictor.

## Immutable validation and safety

- human rows: `365654`
- development rows: `285210`
- sealed gold rows: `80444`
- historical split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`
- canonical row-map SHA: `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`
- dev↔sealed item overlap: `0`
- train↔held item overlap: `0`
- sealed gold opened: `false`
- sealed gold scored: `0`

## LLM-label policy

The legacy weak pool has `11,187,780` rows but exact pair overlap with trusted human labels is `0`, so its class-label precision is not directly auditable. Current new-architecture screens use **zero LLM-labelled rows**. LLM supervision remains denied until an independent human-controlled audit proves it helps.

## New architecture ladder

### A0 — item encoder + component SupCon + bidirectional MaxSim + symmetric head

Run `31883770637`.

- fold0 Macro AP: **`0.5486140975180157`**
- label source: human only
- result: **REJECTED**

Interpretation: independent item encoding plus MaxSim alone loses too much pair-conditioned reasoning relative to the v12 CrossEncoder reference.

### A1 — A0 + deterministic human hard-negative repeats

Same canonical run family.

- fold0 Macro AP: **`0.5422162762826607`**
- delta vs A0: `-0.0063978212353550`
- result: **REJECTED**

Sampler-only hard-negative repetition does not repair the architecture.

### A2 — positive component closure

- status: **cancelled / rejected without metric**
- reason: after A0/A1, spending another full RTX cycle on a sampler-only extension was not credible; the queued job was deliberately retired while preserving A0/A1 evidence.

### A3 — LateInteraction + category MoE + category-local ranking

Run `31887218705`, evidence artifact `9248080049`.

- fold0 Macro AP: **`0.3222800376478955`**
- human-only rows
- optimizer steps: `3803`
- result: **REJECTED**

The category expert/ranking path on top of plain LateInteraction collapsed quality and is closed.

### A5 — compressed pair-conditioned cross interaction

Architecture:

`shared item encoder -> reusable item/token cache -> learned 12-slot compression -> tiny bidirectional cross-attention -> symmetric/category-conditioned head`

This restores pair-conditioned reasoning without returning to a full concatenated pair CrossEncoder. Human-only fold0 run `31891601603` is in progress after fixing a read-only `py_compile` cache issue.

### A8 — Granite-97M + compressed cross interaction

Architecture:

`Granite multilingual retrieval item encoder -> reusable 12-slot item cache -> bidirectional compressed cross-attention -> category MoE/ranking head`

Pinned model:

- `ibm-granite/granite-embedding-97m-multilingual-r2`
- revision `c61e626a6255c490879d0af885078b61929d51f6`
- `model.safetensors` SHA-256 `f3ea88b230492811046145513710e76b4cc8c2ad49e8708da0e7247e548903be`

Two infrastructure failures occurred before any valid quality result: first `py_compile` attempted to write `__pycache__` on a read-only mount; then ModernBERT implicitly invoked TorchInductor/Triton and attempted to write `/root/.triton`. The architecture has now been changed to force `reference_compile=false` in train/production/submission runtime. A fresh A8 quality run is required; the failed runs are **not** quality evidence.

## Prebuilt next candidates if A5/A8 remain too weak

- **A12**: Granite compressed cross-attention + fold-safe typed structured features (`model`, `quantity`, `brand`, same-key attribute conflicts/agreement) fused into one neural head.
- **A6**: LLM-free fold-safe distillation into the compressed-cross student using only candidate pair IDs from the weak retrieval graph; the legacy weak `target` column is not admitted as truth.
- **A10**: stronger `multilingual-e5-base` item encoder with the same compressed-cross head, subject to runtime evidence.

## Promotion rule

Do not run expensive strict OOF or package a model simply because it is architecturally new. A fold0 candidate must first recover to a credible quality region relative to the v12 `0.7059297810` reference. A promoted candidate then must pass:

1. exact five-fold component-disjoint OOF over all `285210` development rows;
2. zero sealed-gold scoring and zero cross-fold item leakage;
3. full-development production refit;
4. one-checkpoint offline package;
5. exact organizer-shaped `<60 s` Check on final ZIP bytes;
6. private Hugging Face upload and byte-for-byte SHA roundtrip.

## Legacy residual fallback

The previously built `ecup-v14-v12-category-gated-residual-submission.zip` remains reproducible and runtime-verified (SHA `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`, organizer-shaped Check `28.810029840000425 s`). It is retained strictly as a fallback/reference and is superseded by the user-requested new-architecture research.
