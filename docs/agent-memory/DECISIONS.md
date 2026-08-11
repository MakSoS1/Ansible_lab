# E-CUP Matching — Durable Decisions

## D001 — Item-disjoint validation is the primary offline gate

**Decision:** Human validation groups connected components of product IDs so no item ID appears in both train and validation.

**Reason:** Hidden evaluation contains new/unseen products. Random pair splits can leak item identity and overstate generalization.

## D002 — Optimize the official category-macro AP, not threshold accuracy

**Decision:** Report AP per category and their unweighted mean. Thresholded accuracy/F1 can be diagnostic but never substitutes for the official metric.

## D003 — v1 uses sklearn HGB rather than CatBoost

**Decision:** First end-to-end anchor uses only dependencies proven in the organizer image.

**Reason:** Runtime probing showed sklearn/joblib are present while CatBoost/RapidFuzz are absent. A small reliable submit is more useful than a fragile package.

## D004 — Selected long-term architecture is a noise-aware distilled hybrid cascade

**Decision:** Combine deterministic product normalization, structured lexical/attribute features, weak-label curriculum/hard negatives and compact neural reranking while keeping organizer runtime explicit.

**Reason:** This gives the strongest quality/runtime/unseen-item trade-off among the researched approaches.

## D005 — Competition artifacts remain private

**Decision:** Raw parquet, models, submit ZIPs and persistent Memora DB never go into public Git. Durable binary artifacts use private HF `Maksim123321/e-cup-2026-matching-private`.

## D006 — Memora is local-only; HF is the persistence layer

**Decision:** Pin Memora to `bc64ff745a9b2c0e6245e0137654f041fba0c155`, harden it, run local SQLite/TF-IDF only, and checkpoint the DB through repository-controlled HF scripts.

**Reason:** This preserves cross-agent semantic memory while removing unnecessary cloud/API/chat attack surface. Markdown in Git stays canonical.

## D007 — New agents recover context from repository state, not hidden chat memory

**Decision:** Root `AGENTS.md` is mandatory. Every retained experiment updates PLAN/RESULTS/index/state and Memora.

**Reason:** GitHub Actions and coding agents are ephemeral; handoff must be deterministic and auditable.

## D008 — v2 first exploits weak labels without increasing inference complexity

**Decision:** Before adding a Transformer to inference, test filtered/confidence-weighted LLM labels and hard negatives on the fast structured anchor.

**Reason:** The 11M soft labels are the largest unused supervision source; if they improve the cheap model, the gain is effectively free at inference time.

## D009 — Retain confidence curriculum; reject static hard-negative reweighting

**Decision:** v2 retains `v2b-weak-curriculum` (Macro AP `0.5010008995`) and rejects the v2c heuristic hard-negative weight boost (`0.4957263069`).

**Reason:** Confidence-filtered weak labels added a gain, while a fixed heuristic hard-negative weight erased it. Future hard negatives should be model-mined examples rather than static multipliers.

## D010 — Do not fabricate neural evidence when GPU allocation fails

**Decision:** Implemented neural paths may remain unretained until a real accelerator run produces comparable metrics.

**Reason:** Earlier Lightning/other GPU allocation probes authenticated but could not allocate usable compute. Infrastructure failure is not a model score.

## D011 — Runtime optimizations must be feature-equivalent

**Decision:** Optimize inference/preprocessing only when regression tests prove equality to the retained recipe.

**Reason:** Runtime changes must not silently redefine a selected model. The accepted single-pass feature optimization and later bounded-memory weak sampling are regression-tested against their previous semantics.

## D012 — Public source executes only through a private isolated GPU dispatcher

**Decision:** Register the home RTX 2060 SUPER only to private `MakSoS1/gpu-dispatch`, never to public `MakSoS1/Ansible_lab`. Public source executes by exact allowed SHA in a network-disabled/read-only container with no repository token, host secret, Docker socket, Windows mount or Linux capability.

**Reason:** A self-hosted runner attached directly to a public repository would turn a malicious workflow/contribution into host code execution. The private dispatcher preserves a trusted orchestration boundary.

## D013 — v4 retains cross-fitted regularized category routing, not the unfinished stronger encoder

**Decision:** Promote v4 by keeping the immutable v3 structured + `rubert-tiny2` models and replacing the single global blend alpha with shrinkage-regularized per-category alphas. Select shrinkage strength only by 5-fold `GroupKFold` over connected components of all validation candidate edges. Retain prior `4000`, whose OOF Macro AP is `0.5276431099433088` versus v3 `0.5254642645846543`.

**Reason:** The routing improvement is measurable without fabricating stronger-encoder evidence, survives component-grouped cross-fitting, and passes exact organizer-image offline execution. The larger `0.5284493942551521` full-data coefficient-fit score is recorded for the packaged coefficients but is not used as the unbiased headline. The canonical v4 ZIP is immutable at SHA-256 `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`.

**Consequence:** v4 is the current submission candidate; v3 remains immutable fallback. The implemented `ai-forever/ruBert-base` v4a/v4b/v4c ladder becomes a future v4.1/v5 ablation and must produce its own real item-disjoint metric before it can replace v4.