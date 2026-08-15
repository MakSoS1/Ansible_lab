# E-CUP v14 — New Architecture Results

Status: **in progress — setwise and token-preserving fold0 screening**

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

The legacy weak pool has `11,187,780` rows but exact pair overlap with trusted human labels is `0`, so its class-label precision is not directly auditable. Current new-architecture screens use **zero LLM-labelled rows**. Retrieval topology may be read without its `target` column. LLM supervision remains denied until an independent human-controlled audit proves it helps.

## Frozen fold0 promotion gate

This rule was recorded before A5 completed:

- `<0.64`: hard quality reject;
- `0.64–<0.68`: research-only, no strict OOF;
- `>=0.68`: credible promotion region; compare surviving screens, then exact five-fold OOF;
- v12 reference: `0.7059297810308699`.

The gate is not relaxed after seeing a result.

## Measured architecture ladder

### A0 — item encoder + component SupCon + bidirectional MaxSim + symmetric head

Run `31883770637`.

- fold0 Macro AP: **`0.5486140975180157`**
- label source: human only
- result: **REJECTED**

Independent item encoding plus MaxSim alone loses too much pair-conditioned reasoning relative to the v12 CrossEncoder reference.

### A1 — A0 + deterministic human hard-negative repeats

- fold0 Macro AP: **`0.5422162762826607`**
- delta vs A0: `-0.0063978212353550`
- result: **REJECTED**

Sampler-only hard-negative repetition does not repair the architecture.

### A2 — positive component closure

- status: **cancelled / rejected without metric**
- reason: after A0/A1, another full RTX cycle on a sampler-only extension was not credible.

### A3 — LateInteraction + category MoE + category-local ranking

Run `31887218705`, evidence artifact `9248080049`.

- fold0 Macro AP: **`0.3222800376478955`**
- optimizer steps: `3803`
- label source: human only
- result: **REJECTED**

Category experts/ranking on top of plain LateInteraction collapsed quality.

### A5 — ruBERT 12-slot compressed pair-conditioned cross interaction

Run `31891601603`, evidence artifact `9249392211`.

Architecture:

`ruBERT item encoder -> 12 learned reusable evidence slots -> bidirectional tiny cross-attention -> symmetric head + category residual experts`, trained with category-local batches and ranking.

- fold0 Macro AP: **`0.3701278186124241`**
- result: **REJECTED** by the predeclared `<0.64` gate
- strict OOF: **not run**

Restoring pair-conditioned attention after aggressive 12-slot compression did not restore quality.

### A8 — Granite-97M 12-slot compressed cross interaction

Run `31891817294`, evidence artifact `9249911397`.

Pinned encoder:

- `ibm-granite/granite-embedding-97m-multilingual-r2`
- revision `c61e626a6255c490879d0af885078b61929d51f6`
- `model.safetensors` SHA-256 `f3ea88b230492811046145513710e76b4cc8c2ad49e8708da0e7247e548903be`
- ModernBERT `reference_compile=false`, so no hidden TorchInductor/Triton path remained in the valid screen.

- fold0 Macro AP: **`0.3450728413820783`**
- result: **REJECTED** by the same frozen gate
- strict OOF: **not run**

Because both ruBERT A5 and Granite A8 fail badly under the same 12-slot + MoE/ranking regime, this family is now rejected independently of backbone choice. Unchanged A10/A12 descendants were suspended instead of spending more GPU hours on the same regime.

## Current pivot

### Contextual setwise architecture

Queued run `31896370680`.

Architecture:

`fold-safe typed features_v2 + six lexical features + category one-hot -> permutation-equivariant Transformer over each anchor candidate set -> bidirectional edge aggregation`.

A pairwise MLP control is trained on the exact same features. The screen reports both absolute Macro AP and `setwise - pairwise` delta, so any value from candidate competition/context is measured separately from the feature set itself. It uses human labels only.

### A6 — fold-safe human-only teacher + LLM-free retrieval targets

Run `31892229544` is currently building:

1. a new human-only ruBERT CrossEncoder on fold-train;
2. held-fold teacher AP;
3. a structured fold-train-only teacher;
4. up to `350k` retrieval-graph pairs whose legacy LLM `target` column is never read;
5. soft labels from `0.8 * neural + 0.2 * structured`, with teacher disagreement reducing example weight.

This run is teacher/data construction, not a promoted student by itself.

### A15 — token-preserving pair-conditioned architecture

Queued run `31897786692`, conditional on the A6 human-only teacher reaching at least `0.64` held-fold AP.

Architecture:

`ruBERT item encoder -> actual projected title/attribute token evidence (16 title + 32 attribute tokens; no learned slot averaging) -> shared bidirectional masked cross-attention -> fold-train-only typed features_v2 + category one-hot -> global symmetric head`.

Training deliberately removes the A3/A5 confounders:

- global inverse-category sampling, not single-category batches;
- BCE + small pair contrastive term;
- ranking weight `0`;
- no category experts;
- no LLM labels.

A base-init A15 fallback workflow is prepared but not queued; it is used only if the A6 teacher fails its admission floor.

## Promotion path

No new architecture is packaged merely for novelty. A fold0 survivor at `>=0.68` must then pass:

1. exact five-fold component-disjoint OOF over all `285210` development rows;
2. fold-specific teacher training where required, so no held-fold label reaches warm-start training;
3. zero sealed-gold scoring and zero cross-fold item leakage;
4. full-development production refit;
5. one-checkpoint offline package;
6. exact organizer-shaped `<60 s` Check on final ZIP bytes;
7. private Hugging Face upload and byte-for-byte SHA roundtrip.

## Legacy residual fallback

The previously built `ecup-v14-v12-category-gated-residual-submission.zip` remains reproducible and runtime-verified (SHA `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`, organizer-shaped Check `28.810029840000425 s`). It is retained strictly as a fallback/reference and is superseded by the user-requested new-architecture research.
