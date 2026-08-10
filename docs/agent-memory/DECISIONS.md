# E-CUP Matching — Durable Decisions

## D001 — Item-disjoint validation is the primary offline gate

**Decision:** Human validation groups connected components of product IDs so no item ID appears in both train and validation.

**Reason:** Hidden evaluation contains new/unseen products. Random pair splits can leak item identity and overstate generalization.

## D002 — Optimize the official category-macro AP, not threshold accuracy

**Decision:** Report AP per category and their unweighted mean. Thresholded accuracy/F1 can be diagnostic but never substitutes for the official metric.

## D003 — v1 uses sklearn HGB rather than CatBoost

**Decision:** First end-to-end anchor uses only dependencies proven in the organizer image.

**Reason:** Runtime probing showed sklearn 1.9.0/joblib 1.4.2 are present while CatBoost/RapidFuzz are absent. A small reliable submit is more useful than a fragile package.

## D004 — Selected long-term architecture is a noise-aware distilled hybrid cascade

**Decision:** Combine deterministic product normalization, structured lexical/attribute features, a multilingual bi-encoder, weak-label curriculum/hard negatives, compact distilled Cross-Encoder, and uncertainty gating.

**Reason:** Best quality/runtime/unseen-item trade-off among the 10 researched approaches.

## D005 — Competition artifacts remain private

**Decision:** Raw parquet, models, submit ZIPs and persistent Memora DB never go into public Git. Durable binary artifacts use private HF `Maksim123321/e-cup-2026-matching-private`.

## D006 — Memora is local-only; HF is the persistence layer

**Decision:** Pin Memora to `bc64ff745a9b2c0e6245e0137654f041fba0c155`, harden it, run local SQLite/TF-IDF only, and checkpoint the DB through repository-controlled HF scripts.

**Reason:** This preserves cross-agent semantic memory while removing unnecessary Cloudflare/API/chat attack surface. Markdown in Git stays canonical so the project remains understandable without MCP.

## D007 — New agents must recover context from repository state, not hidden chat memory

**Decision:** Root `AGENTS.md` is mandatory. Client-specific files only redirect to it. Every retained experiment updates PLAN/RESULTS/index/state and Memora.

**Reason:** GitHub Actions and coding agents are ephemeral; handoff must be deterministic and auditable.

## D008 — v2 first exploits weak labels without increasing inference complexity

**Decision:** Before adding a Transformer to inference, test filtered/confidence-weighted LLM labels and hard negatives on the fast structured anchor.

**Reason:** The 11M soft labels are the largest unused supervision source; if they improve the cheap model, the gain is effectively free at inference time.

## D009 — Retain confidence curriculum; reject static hard-negative reweighting

**Decision:** v2 retains `v2b-weak-curriculum` (Macro AP `0.5010008995`) and rejects the v2c heuristic hard-negative weight boost (`0.4957263069`).

**Reason:** 2024-inspired product-aware features improved the fixed item-disjoint score, and 300k confidence-filtered weak labels added another small gain. A fixed heuristic hard-negative weight erased the gain. Future hard negatives must therefore be **model-mined examples** for a reranker/second-stage training curriculum rather than static sample-weight multipliers.

## D010 — GPU reranker is deferred until Lightning exposes an accessible Studio

**Decision:** Keep the implemented RuBERT-tiny2 soft-label reranker, model-driven hard-negative stage and secure Lightning bridge, but do not fabricate a neural metric or make it part of v2.

**Reason:** Current Lightning authentication succeeds and Teamspace discovery works, but the account exposes no reusable Studio through the SDK and denies `create_cloud_space` with HTTP 403. No GPU training actually started. v2 therefore ships the verified structured model, while v3 can resume the neural path as soon as a Studio exists.

## D011 — Runtime optimizations must be feature-equivalent

**Decision:** Optimize the structured v2 inference implementation only when regression tests prove equality to the selected model's original feature vector.

**Reason:** The first 275k organizer-image benchmark was close to the private runtime limit. The accepted single-pass symmetry optimization replaces a duplicated full `_pair_features` pass with reverse evaluation of only the directional fuzzy features and is regression-tested against the previous vector at `1e-15` tolerance. This reduces runtime risk without retraining or changing validation predictions.
