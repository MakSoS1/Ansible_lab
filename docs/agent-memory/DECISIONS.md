# E-CUP Matching — Durable Decisions

## D001 — Item-disjoint validation is the primary offline gate

**Decision:** Human validation groups connected components of product IDs so no item ID appears in both train and validation.

**Reason:** Hidden evaluation contains new/unseen products. Random pair splits can leak item identity and overstate generalization.

## D002 — Optimize the official category-macro AP, not threshold accuracy

**Decision:** Report AP per category and their unweighted mean. Thresholded accuracy/F1 can be diagnostic but never substitutes for the official metric.

## D003 — v1 uses sklearn HGB rather than CatBoost

**Decision:** First end-to-end anchor uses only dependencies proven in the organizer image.

## D004 — Selected long-term architecture is a noise-aware distilled hybrid cascade

**Decision:** Combine deterministic product normalization, structured lexical/attribute features, weak-label curriculum/hard negatives and compact neural reranking while keeping organizer runtime explicit.

## D005 — Competition artifacts remain private

**Decision:** Raw parquet, models, submit ZIPs and persistent Memora DB never go into public Git. Durable binary artifacts use private HF `Maksim123321/e-cup-2026-matching-private`.

## D006 — Memora is local-only; HF is the persistence layer

**Decision:** Pin Memora to `bc64ff745a9b2c0e6245e0137654f041fba0c155`, run local SQLite/TF-IDF only, and checkpoint the DB through repository-controlled HF scripts.

## D007 — New agents recover context from repository state, not hidden chat memory

**Decision:** Root `AGENTS.md` is mandatory. Every retained experiment updates PLAN/RESULTS/index/state and Memora.

## D008 — v2 first exploits weak labels without increasing inference complexity

**Decision:** Before adding a Transformer to inference, test filtered/confidence-weighted LLM labels and hard negatives on the fast structured anchor.

## D009 — Retain confidence curriculum; reject static hard-negative reweighting

**Decision:** v2 retains `v2b-weak-curriculum` (`0.5010008995`) and rejects v2c static hard-negative weight boost (`0.4957263069`).

## D010 — Do not fabricate neural evidence when GPU allocation fails

**Decision:** Implemented neural paths may remain unretained until a real accelerator run produces comparable metrics.

## D011 — Runtime optimizations must be feature-equivalent

**Decision:** Optimize inference/preprocessing only when regression tests prove equality to the retained recipe.

## D012 — Public source executes only through a private isolated GPU dispatcher

**Decision:** Register the home RTX 2060 SUPER only to private `MakSoS1/gpu-dispatch`, never to public `MakSoS1/Ansible_lab`. Public source executes by exact allowed SHA in a network-disabled/read-only container with no repository token, host secret, Docker socket, Windows mount or Linux capability.

## D013 — v4 retains cross-fitted regularized category routing, not the unfinished stronger encoder

**Decision:** Historical v4 keeps immutable v3 models and replaces global blend alpha with shrinkage-regularized per-category alphas selected only by component-grouped crossfit. OOF `0.5276431099433088`; canonical SHA `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`.

**Consequence:** Later hidden evidence supersedes v4 as a production-selection conclusion; see D014 onward.

## D014 — Hidden leaderboard evidence makes v2 the production anchor for v5 development

**Decision:** Until v5 passes a new sealed evaluation, use v2 as the production/leaderboard fallback.

**Evidence:** hidden Macro AP: v1 `0.23458522924335687`, v2 `0.2583231811423486`, v3 non-canonical `0.2583231811423486`, v3 canonical `0.24810151893254498`, v4 canonical `0.2531285194869718`.

**Consequence:** “production best” and “development CV best” are separate fields.

## D015 — v5 uses one immutable five-fold development split plus sealed gold

**Decision:** Freeze `285,210` development rows, `80,444` sealed-gold rows, five folds, zero cross-split item overlap, SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.

**Rules:** Gold labels are not read during development; gold items are excluded from representation adaptation/mining; candidate/config/preprocessing hashes freeze before one-shot gold.

## D016 — Category-specific structured models are the retained v5 structured base

**Decision:** One compact HGB per category replaces the global HGB as v5 structured base.

**Evidence:** audit OOF `0.5315527708634168`; category-specialist OOF `0.5476780661335778`, delta `+0.016125295270161044`, all five folds improve.

## D017 — Reject direct attribute likelihood score shifts

**Decision:** Do not directly add category/key log-likelihood evidence to base logits or tune a rescue scalar on the same folds.

**Evidence:** OOF `0.523218903672764`, delta `-0.008333867190652766`, all five folds regress.

## D018 — Pretrained item embeddings alone are insufficient

**Decision:** Ready-made multilingual bi-encoder without task supervision is not retained standalone.

**Evidence:** stacked OOF `0.5318080650341337`, only `+0.0002552941707169021` over audit base; raw cosine AP ~`0.3120`.

## D019 — Leakage-safe weak category specialists are retained development evidence

**Decision:** Keep weak-label specialist curriculum only when each outer fold excludes held-fold and sealed-gold items from weak sampling.

**Evidence:** OOF `0.5514237338676234`, delta `+0.00374566773404561` vs category base; all five folds improve. Run `31484641329`.

## D020 — Cross-fitted OOF stacking may retain individually weak signals only when the stack improves every fold

**Decision:** Keep the v5 combo of already-OOF category, weighted and pretrained semantic inputs when the second level is itself cross-fitted.

**Evidence:** combo OOF `0.559512531439709`, delta `+0.011834465306131192` vs category base; all five folds improve.

**Consequence:** Weighted specialists and pretrained embeddings are not promoted standalone, but their OOF information may survive inside a separately cross-fitted stack.

## D021 — Memora checkpoints are created only from GREEN repository state

**Decision:** Keep full repository tests and memory-policy gates before ingest. Never weaken them to make a checkpoint pass during intentionally RED TDD.

**Incident:** runs `31481012401` and `31482891498` failed before ingest. In `31482891498`, collection failed on missing `ecup_matching.ml.v5_weighted_specialists`; later GREEN code did not retroactively create the checkpoint.

**Protocol:** A result is safely handed off only after tests, `memory_policy.py`, Memora ingest, SQLite integrity/secret scan, private HF upload and remote verification all pass.

## D022 — Infrastructure failures are not model-quality failures

**Decision:** Record resource/integration failures separately from hypothesis outcomes.

**Examples:** initial contrastive physical batch 96 OOMed on MPS; microbatch 24 × accumulation 4 preserved effective 96. Earlier v4 RTX run exited 137 before metrics. Neither is a model score.

## D023 — Supervised contrastive item-space is retained and is the current v5 development best

**Decision:** Retain supervised contrastive item embeddings as a major v5 signal and use their stacked OOF `0.5662217062664492` as the current development benchmark.

**Evidence:** category base `0.5476780661335778`; delta `+0.018543640132871353`; all five held folds improve to `0.5692965046798911`, `0.5683388560864314`, `0.5694312406050667`, `0.5632500994392833`, `0.5684466083884651`. Raw supervised semantic cosine AP rises to `0.40597111640267125`. Run `31483288887`, source `b30821f613bf7051da51c42b64c7f79361d5619c`.

**Consequence:** The pretrained bi-encoder failure to add much signal did not falsify item-space learning; task supervision materially changes the representation. Future neural work must compare incrementally against `0.5662217062664492` or a later stronger honest OOF benchmark.

## D024 — Strict train-only sparse TF-IDF is retained as an independent strong signal

**Decision:** Keep sparse rare-token/model-code similarity when each outer fold fits vocabulary/IDF only on outer-train items and transforms held items without refitting.

**Evidence:** OOF `0.5651306838802859`, delta `+0.017452617746708032` vs category base, all five folds improve. Run `31485396599`, source `634ee66890c39ad97c0fa725135b1b00e56ac126`.

**Consequence:** Rare model/SKU token weighting is a real transferable signal and is a prime candidate for a leakage-safe combination with supervised contrastive predictions; do not replace it with global TF-IDF fit over held/gold items.

## D025 — Heavy workflow helper APIs require integration tests; first ruBERT teacher run is not a model rejection

**Decision:** Do not infer model quality from workflow `31485127564`. It failed before comparable predictions because `train_v5_teacher_fold.py` called `build_reranker_examples(items, curriculum)` after the helper required an `attribute_importance` argument.

**Evidence:** all five teacher fold jobs failed with `TypeError: build_reranker_examples() missing 1 required positional argument: 'attribute_importance'`; aggregate was skipped.

**Consequence:** Fix the call explicitly and cover the real integration boundary in tests before rerunning. A unit test that validates helpers/MPS independently is insufficient when a heavy workflow depends on their composed call signature.

## D026 — Machine-readable current/safe metrics are first-class Memora sources

**Decision:** `ecup_matching/experiments/CURRENT.json` and every `ecup_matching/experiments/v*/SAFE_METRICS.json` belong in `scripts.memory_ingest.py::canonical_sources()` alongside durable Markdown, PLAN and RESULTS.

**Reason:** Before this audit, those machine-readable files could be correct in Git yet invisible to semantic Memora retrieval. A regression test now requires their inclusion.
