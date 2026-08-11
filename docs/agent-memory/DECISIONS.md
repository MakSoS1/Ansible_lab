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

**Consequence:** v4 was the retained offline candidate at the end of v4. Later hidden leaderboard evidence supersedes this as a production-selection conclusion; see D014 onward.

## D014 — Hidden leaderboard evidence makes v2 the production anchor for v5 development

**Decision:** Until v5 passes a new sealed evaluation, use v2 as the production/leaderboard fallback. Keep v3/v4 artifacts immutable but treat their larger old local scores as historical offline evidence, not as proof of better hidden transfer.

**Evidence:** observed hidden Macro AP: v1 `0.23458522924335687`, v2 `0.2583231811423486`, v3 non-canonical `0.2583231811423486`, v3 canonical `0.24810151893254498`, v4 canonical `0.2531285194869718`.

**Consequence:** “production best” and “development CV best” are separate fields and must never be collapsed into one ambiguous “best model”.

## D015 — v5 uses one immutable five-fold development split plus sealed gold

**Decision:** Freeze the v5 split before further selection: `285,210` development rows, `80,444` sealed-gold rows, 5 development folds, zero cross-split item overlap, split SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.

**Rules:** Sealed-gold labels are not read during development. Sealed-gold items are also excluded from representation adaptation, weak mining and hard-negative mining. Candidate/config/preprocessing hashes are frozen before the one-shot gold gate.

**Reason:** Repeatedly optimizing the old 73,131-row holdout produced local improvements that did not transfer monotonically to hidden evaluation. The new protocol makes the requested `0.60` target an honest development-OOF target rather than a tunable holdout number.

## D016 — Category-specific structured models are the retained v5 structured base

**Decision:** Train one compact HGB per category rather than a single global HGB shared across 20 heterogeneous product regimes.

**Evidence:** audit baseline OOF `0.5315527708634168`; category-specialist OOF `0.5476780661335778`, delta `+0.016125295270161044`, with improvement on all five held folds.

**Consequence:** New semantic/neural/weak branches must prove incremental OOF value against the category-specialist base, not only against the weaker global baseline.

## D017 — Reject direct attribute likelihood score shifts

**Decision:** Do not directly add category/key log-likelihood evidence to the base logit and do not rescue it by fitting a post-hoc scalar on the same development folds.

**Evidence:** OOF fell from `0.5315527708634168` to `0.523218903672764` (`-0.008333867190652766`) and every fold regressed.

**Consequence:** Attribute knowledge may be exposed as normal features for the estimator, where the model can condition its use, rather than as an unconditional score shift.

## D018 — Pretrained item embeddings alone are insufficient

**Decision:** A ready-made multilingual bi-encoder without task supervision is not retained standalone.

**Evidence:** stacked OOF `0.5318080650341337`, only `+0.0002552941707169021` over the audit base; raw semantic cosine Macro AP is about `0.3120`.

**Consequence:** Neural item-space work must use supervised/weak/domain adaptation and demonstrate OOF gain over the retained category-specialist base.

## D019 — Leakage-safe weak category specialists are retained development evidence

**Decision:** Keep the weak-label specialist curriculum when each outer fold excludes held-fold and sealed-gold items from weak sampling.

**Evidence:** category base `0.5476780661335778`; weak-specialist OOF **`0.5514237338676234`**, delta `+0.00374566773404561`; all five folds improved. Run `31484641329`, source `319993a469cfa37770d66cfaf1b2203515dc9841`.

**Consequence:** Weak labels remain useful, but only under item-exclusion checks and fold-specific evaluation. Do not infer that “more weak rows” is automatically better; scale only after OOF evidence.

## D020 — Cross-fitted OOF stacking may retain individually weak signals only when the stack improves every fold

**Decision:** Keep the v5 combo of category-specialist, fold-weighted and pretrained-semantic OOF inputs as the current development benchmark. The second-level estimator must itself be trained cross-fold so a row never trains the stack on its own target/in-sample base prediction.

**Evidence:** combo OOF **`0.559512531439709`** vs category base `0.5476780661335778`, delta `+0.011834465306131192`. Held-fold AP is `0.562580065789817`, `0.5605549739646596`, `0.5667063354890245`, `0.5579194708823281`, `0.5631351691480011`; all five folds improve.

**Consequence:** The standalone fold-weighted model (`0.5498696731704964`) is not promoted because two folds regress, and the pretrained encoder is not promoted because its gain is tiny; nevertheless their already-OOF information may be used in a separately cross-fitted stack when the combined result is consistently better.

## D021 — Memora checkpoints are created only from GREEN repository state

**Decision:** Keep the full repository test and memory-policy gates before Memora ingest. Do not weaken them to make a memory checkpoint pass during an intentionally RED TDD phase.

**Incident:** v5 memory runs `31481012401` and `31482891498` failed before ingest. In `31482891498`, test collection failed with `ModuleNotFoundError: ecup_matching.ml.v5_weighted_specialists` because the documentation update landed while the next TDD test was intentionally RED. Production code later made the workspace GREEN, but the skipped checkpoint was never retroactively created.

**Protocol:** Update durable KEEP/REJECT facts after the relevant TDD cycle is GREEN. If a memory-doc commit was made while RED, a later GREEN commit must touch a memory-triggering file or manually dispatch `ecup-memora-memory.yml`. A result is not considered safely handed off until full tests, `memory_policy.py`, Memora ingest, SQLite integrity/secret scan, private HF upload and remote verification all pass.

## D022 — Infrastructure failures are not model-quality failures

**Decision:** Record and fix resource failures separately from hypothesis outcomes.

**Examples:** v5 contrastive training initially OOMed on MPS with physical batch 96; the correction uses microbatch 24 with gradient accumulation 4 to preserve effective batch 96. The earlier v4 `ruBert-base` RTX run died with exit 137 before metrics due host-memory-heavy preprocessing and model load order. Neither event is a negative model score.

**Consequence:** Never mark a modeling hypothesis REJECT solely because a run failed before producing comparable held-fold predictions.
