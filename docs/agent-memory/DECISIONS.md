# E-CUP Matching — Durable Decisions

This file is a compact durable decision log. Detailed historical evidence remains in iteration RESULTS files and Git history.

## D001–D027 — Foundational rules retained

- Primary validation is item/component-disjoint; official metric is unweighted Macro AP across 20 categories.
- Competition data/models/OOF/submission ZIPs and Memora DB stay private; public Git contains source and aggregate evidence only.
- Hardened Memora is local SQLite/TF-IDF with pinned upstream commit; private HF is persistence.
- Public source never executes directly on the home RTX runner; GPU execution goes through private `gpu-dispatch`.
- Infrastructure failures are not model-quality failures.
- v5 immutable split is `285,210` dev + `80,444` sealed gold, 5 folds, zero item overlap, SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed gold remains one-shot and unopened during development.
- Weak, sparse, supervised-contrastive and explicit per-key signals all produced real leakage-safe local gains; direct unconditional attribute-likelihood shifts were rejected.
- Target-fitted meta layers must be genuinely outer-cross-fitted.
- Memora checkpoints are created only from GREEN repository state, and `CURRENT.json` plus every `SAFE_METRICS.json` are first-class memory sources.

## D028 — v5 six-signal evidence is useful but runtime cost matters

The six-signal stack reached strict OOF `0.5975445721`; later category-shrunk/meta variants crossed `0.60`, including category-shrunk + HGB equal-rank fusion `0.6018115534`. These experiments prove additional structured/sparse/neural signals contain information, but they do not justify paying their full cost at submission inference.

## D029 — Runtime optimization must preserve prediction semantics

Parallelization/cache changes are acceptable only with feature/prediction equivalence evidence. Large-N workload measurements, not tiny smoke tests, are required for runtime claims.

## D030 — v10/v11 runtime architecture is closed

The structured/TF-IDF/multi-model branch remained CPU/startup sensitive and ultimately timed out on platform. The historical local fixture used training `pairs.head(N)`-style data and never exercised an exact Check-sized stage, so its apparent headroom was not reliable.

## D031 — Exact Check-stage startup cost is a first-class gate

Forensic run `31789001358` used historical v11 with organizer image and full 4.1-GB item universe. It reached `60.033 s` without a valid output. Heavy item scans/deserialization can therefore fail before per-pair throughput matters.

## D032 — One CrossEncoder is the default submission architecture

Starting from v7, keep one tokenizer + one compact ruBERT pair CrossEncoder + one checkpoint. Structured/contrastive/LLM/teacher complexity should be moved offline into training/distillation unless an added inference branch independently clears conservative runtime gates.

## D033 — External leaderboard evidence overrides local-score storytelling

Authoritative anchors: v7 Public LB `0.3655833314`, v12 Public LB `0.3798116204`. `ΔLB=+0.0142282890`; comparable local fold0 delta `+0.0035495184`. Local fold0/OOF magnitude is not a leaderboard estimate.

## D034 — Validation v3 is directional, retrieval-hard and human-truth-only

Use component-disjoint human truth, category-local prevalence/hardness/degree strata and deterministic stress replicas. LLM extreme pseudo-labels may be diagnostics but are not validation truth. Do not fit a linear local→LB calibration from two anchors.

## D035 — Prevalence alone does not explain the local→LB gap

Canonical distribution run `31788445849` measured `365,654` human and `11,187,780` weak rows. Human prevalence `0.2567727961` and weak soft-target mean `0.2435606726` are too similar to explain the gap. Weak candidate degree/hardness is materially different and becomes a primary validation/training axis.

## D036 — Graph post-processing is rejected on quality, not runtime

Predeclared target-free graph variants were backtested on aligned v7/v8/v12 fold0 predictions. Best variant changed AP by v7 `-0.00018495`, v8 `+0.00010880`, v12 `-0.00014803`. Runtime (~`1.29 s / 275k`) is safe, but quality stability is absent. Do not tune graph weights post hoc to v12.

## D037 — Preserve retrieval-anchor topology before changing the loss

Source audit found v7/v12 weak sampling split candidate groups and canonical pair ordering destroyed the original retrieval-anchor orientation. v13 B therefore changes only data topology/orientation while preserving model, weak exposure and runtime.

## D038 — Groupweak B is locally positive but is not yet a leaderboard keeper

Run `31791177120`: fold0 `0.7086611385531062` versus v12 `0.7059297810308699`, delta `+0.0027313575222363`. This is sufficient to package B as the next external candidate, not sufficient to claim Public-LB improvement.

## D039 — More sophisticated ranking loss must earn its place

Controlled equal-exposure C2/ListNet regressed the relevant diagnostics and was rejected. Complexity is not retained by default. Ambiguous all-soft/listwise experiments remain separate causal ablations.

## D040 — Runtime acceptance distinguishes binding subset Check from full-item stress

For v13 B, the binding organizer-shaped supplied-item Check (1,000 rows, 1,999 materialized items) passed in `26.135347286995966 s`, with valid output and return code 0. A deliberately stricter diagnostic that scans the entire `4,104,103,411`-byte item universe timed out at `60.004995396971935 s`.

The first result is the binding candidate acceptance under the current supplied-item subset contract; the second remains an explicit residual-risk diagnostic. Never quote one as the other.

## D041 — Artifact identity is cryptographic

The exact v13 B candidate is `ecup-v13-groupweak-v7runtime-submission.zip`, `663760087` bytes, SHA-256 `f4b7aad36c8d293a3939d9fb2ce7f91cff1bd8381c870015b2f16ea65a17badb`. Friendly version names are insufficient for submission provenance.

## D042 — Private-HF publication requires download-back verification

HF run `31843423348` uploaded the candidate, downloaded the exact path back, checked bytes and SHA, and produced `canonical.zip: OK` plus `V13_CANDIDATE_HF_ROUNDTRIP_VERIFIED`. GitHub Releases are not part of the final artifact path.

## D043 — v13 B remains a candidate until platform evidence exists

`strict_five_fold_confirmed=false` and `strict_final_keeper_claimed=false` are deliberate. v12 `0.3798116204` remains the best observed Public-LB anchor until the exact v13 B archive receives a platform score.

## D044 — Package reproduction is bound to exact source provenance

The package builder used public source commit `4e83294eb5f6c31c720f7cbb0220f0f4d0ee3cb1`. The later `ecup-matching-2026` branch tip diverged while being used for the one-time HF bridge. Reproduce the packaged runtime from the exact source SHA; do not silently substitute current branch-tip code.

## D045 — v13 is a measured negative external anchor

The exact v13 B archive returned Public LB `0.3783781653`, below v12 `0.3798116204` by `0.0014334551`, despite v13 fold0 `0.7086611385531062` being above v12 `0.7059297810308699`. This is a real local/external ordering inversion. Validation-v3/fold0 may remain diagnostics, but they cannot by themselves promote near-neighbour candidates.

## D046 — Historical split identity is the row map, not a newly recomputed hash

Rebuilding the split in the current environment produced a different manifest hash. v14 therefore recovered the original development/sealed/fold row map from historical strict OOF artifact `9175469673` / run `31680767570` and pinned it by SHA-256 `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`.

The recovered map was applied to the current `matches.parquet` and independently verified to have zero dev↔sealed item overlap and zero train↔held item overlap in all five folds. Never silently accept a recomputed replacement split as "the same" validation.

## D047 — The historical LLM weak stream is not admitted without ground-truth overlap

`matches_llm.parquet` contains `11,187,780` rows but exact pair overlap with human truth is `0`. Positive/negative precision therefore cannot be measured on controlled human labels. Confidence or target extremeness is not a substitute for an audited label stream. v14 uses zero LLM-labelled rows.

## D048 — Expensive A2 research is not a quality rejection

The v14 item-centric / late-interaction A2 branch fixed two engineering issues: endpoint-only-as-`id2` component closure and canonical split wiring. A complete new-Transformer five-fold was stopped after projected multi-hour RTX cost. Record A2 as unfinished architectural research, not as a failed-quality candidate.

## D049 — Residual promotion must be cross-fit and decision-rule-consistent

The scalar v1 residual was rejected: fold0 delta `+0.0001070251`, cross-fit mean `+0.0002866901`, only 11/20 categories non-negative.

The first category-gated v2 aggregate evaluator had a bug: it applied opposite-half alpha even to categories that the category gate later rejected. That evaluator was corrected so rejected categories remain exactly at v12 base. Promotion thresholds were not loosened.

Corrected run `31882322590` passed the frozen rule with fold0 `0.7065769713851786`, delta `+0.0006471903543086022`, side deltas `+0.000437006267165585` / `+0.000734831086673049`, cross-fit mean `+0.000585918676919317`, six admitted categories, and 20/20 full-fold categories non-negative.

## D050 — historical residual package is a reproducible fallback

The package formerly called the v14 final contains one v12 ruBERT checkpoint plus a small category-specific six-feature lexical residual trained on human development labels only. It remains reproducible as `ecup-v14-v12-category-gated-residual-submission.zip`, `663770301` bytes, SHA-256 `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`, and passed the organizer-shaped Check in `28.810029840000425 s / 60 s` with `910` unique scores.

This is historical technical evidence only. It was superseded when v14 was explicitly redirected to a genuinely new architecture; it must not be presented as completion of the active v14 research.

## D051 — active v14 requires a genuinely new pair-conditioned item-centric architecture

A0 item-centric LateInteraction (`0.5486140975180157`), A1 plus human hard repeats (`0.5422162762826607`) and A3 LateInteraction + category MoE/ranking (`0.3222800376478955`) established that independent item scoring loses too much pair-specific information. The active architecture therefore encodes each unique item independently but restores pair-conditioned reasoning after learned token compression using a tiny bidirectional cross-attention block.

The active screen ladder is A5 ruBERT compressed-cross, A8 Granite-97M compressed-cross, A5c clean compressed-cross without category experts/ranking, then A12 typed structured fusion, A6 LLM-free retrieval-domain distillation and A10 multilingual-E5 reserve as needed. Legacy LLM targets are not admitted; A6 reads retrieval pair IDs only and generates new soft targets from fold-safe human-trained teachers.

The fold0 promotion rule was frozen before the A5 result: `<0.64` is an architectural reject; `0.64–<0.68` remains research-only; `>=0.68` enters the credible strict-OOF region. v12 fold0 `0.7059297810308699` remains the local reference, but no local score is a Public-LB estimate. Any final candidate additionally requires exact five-fold OOF, full-development refit, an exact-byte organizer Check `<60 s`, and private-HF download-back SHA verification before it can be called submission-ready.
