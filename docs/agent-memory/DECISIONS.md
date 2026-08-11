# E-CUP Matching — Durable Decisions

## D001 — Item-disjoint validation is the primary offline gate

**Decision:** Human validation groups connected components of product IDs so no item ID appears in both train and validation.

## D002 — Optimize official category Macro AP

**Decision:** Report AP per category and their unweighted mean. Threshold accuracy/F1 is diagnostic only.

## D003 — v1 uses organizer-proven sklearn HGB

**Decision:** First anchor uses dependencies known to exist in the organizer image.

## D004 — Long-term architecture is a noise-aware hybrid cascade

**Decision:** Combine deterministic normalization, structured lexical/attribute features, weak-label curriculum/hard negatives and compact neural signals while keeping runtime explicit.

## D005 — Competition artifacts remain private

**Decision:** Raw parquet, models, submit ZIPs, OOF predictions and Memora DB never enter public Git. Durable binary artifacts use private HF `Maksim123321/e-cup-2026-matching-private`.

## D006 — Memora is local-only; HF is persistence

**Decision:** Pin Memora to `bc64ff745a9b2c0e6245e0137654f041fba0c155`, use local SQLite/TF-IDF only, and checkpoint through repository-controlled HF scripts.

## D007 — Agents recover from repository state, not hidden chat memory

**Decision:** Root `AGENTS.md` is mandatory. Every retained experiment updates PLAN/RESULTS/index/state and Memora.

## D008 — v2 exploits weak labels before increasing inference complexity

**Decision:** Test filtered/confidence-weighted LLM labels and hard negatives on the fast structured anchor before Transformer inference.

## D009 — Keep weak confidence curriculum; reject static hard-negative reweighting

**Decision:** v2 retains `v2b-weak-curriculum` (`0.5010008995`) and rejects v2c static hard-negative reweighting (`0.4957263069`).

## D010 — Do not fabricate neural evidence when infrastructure fails

**Decision:** Implemented neural paths remain unretained until a real accelerator run produces comparable metrics.

## D011 — Runtime optimizations must be feature-equivalent

**Decision:** Optimize preprocessing/inference only with regression evidence that retained semantics are unchanged.

## D012 — Public source executes only through private isolated GPU dispatch

**Decision:** Home RTX runner belongs only to private `MakSoS1/gpu-dispatch`, never directly to public `Ansible_lab`.

## D013 — Historical v4 retains cross-fitted category routing

**Decision:** v4 historical OOF `0.5276431099433088`; canonical SHA `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`. Later hidden evidence supersedes it as production-selection evidence.

## D014 — Hidden evidence makes v2 the production anchor during v5

**Decision:** Until v5 passes sealed evaluation, v2 is production/leaderboard fallback.

**Evidence:** hidden AP: v1 `0.23458522924335687`, v2 `0.2583231811423486`, v3 noncanonical `0.2583231811423486`, v3 canonical `0.24810151893254498`, v4 canonical `0.2531285194869718`.

**Consequence:** Production best and development best are separate concepts.

## D015 — v5 uses one immutable five-fold development split plus sealed gold

**Decision:** Freeze `285,210` dev rows, `80,444` sealed-gold rows, 5 folds, zero overlap, SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.

**Rules:** No gold labels/items during development; freeze candidate/config/preprocessing before one-shot gold.

## D016 — Category-specific HGB is the retained v5 structured base

**Evidence:** audit `0.5315527708634168`; category specialists `0.5476780661335778`, +`0.016125295270161044`, all folds improve.

## D017 — Reject direct attribute likelihood score shifts

**Evidence:** OOF `0.523218903672764`, -`0.008333867190652766`, all folds regress.

**Consequence:** Key-specific attribute knowledge should enter as estimator features, not unconditional score shifts.

## D018 — Pretrained item embeddings alone are insufficient

**Evidence:** stack `0.5318080650341337`, only +`0.0002552941707169021`; raw cosine ~`0.3120`.

## D019 — Leakage-safe weak category specialists are retained

**Evidence:** OOF `0.5514237338676234`, +`0.00374566773404561` vs category base, all folds improve; held/gold items excluded. Run `31484641329`.

## D020 — Cross-fitted stacking may use weak standalone signals only with honest OOF gain

**Evidence:** category+weighted+pretrained combo `0.559512531439709`, +`0.011834465306131192`, all folds improve.

## D021 — Memora checkpoints are created only from GREEN repository state

**Decision:** Full tests and memory policy remain hard gates before ingest/checkpoint.

**Incident:** runs `31481012401` and `31482891498` failed before ingest because memory-triggering commits landed during intentionally RED TDD; later GREEN code did not retroactively checkpoint.

## D022 — Infrastructure failures are not model-quality failures

**Examples:** contrastive physical batch 96 MPS OOM; historical RTX exit 137. Diagnose/fix before interpreting model quality.

## D023 — Supervised contrastive item-space is retained

**Evidence:** OOF `0.5662217062664492`, +`0.018543640132871353` vs category base; all 5 folds improve; raw semantic cosine `0.40597111640267125`. Run `31483288887`.

**Consequence:** Task supervision, not merely pretrained embeddings, creates material semantic value.

## D024 — Strict train-only sparse TF-IDF is retained

**Evidence:** OOF `0.5651306838802859`, +`0.017452617746708032`, all folds improve. Run `31485396599`.

**Consequence:** Rare model/SKU token weighting is a strong orthogonal signal; vocabulary/IDF must remain outer-train-only.

## D025 — Heavy workflow helper APIs require integration tests

**Decision:** First ruBERT teacher run `31485127564` is not a model rejection.

**Evidence:** all folds failed before predictions because `build_reranker_examples` was called without required `attribute_importance`.

**Consequence:** Fix the composed integration call and test it before rerun; never fabricate a teacher AP from failure.

## D026 — Machine-readable current/safe metrics are first-class Memora sources

**Decision:** `ecup_matching/experiments/CURRENT.json` and `ecup_matching/experiments/v*/SAFE_METRICS.json` must be part of `scripts.memory_ingest.py::canonical_sources()`.

**Reason:** Before this audit those files could be correct in Git but invisible to semantic Memora retrieval. A regression test requires their inclusion.

## D027 — Explicit per-key attribute features are retained and are current v5 dev best

**Decision:** Keep fold-trained explicit per-key attribute match/conflict/missing features inside category specialists. This is distinct from rejected direct likelihood score shifting: the estimator decides when a key matters instead of receiving an unconditional logit correction.

**Evidence:** OOF **`0.5683065131240066`** vs category base `0.5476780661335778`, delta **`+0.02062844699042876`**. Held folds: `0.5706378464826163`, `0.5682631251392076`, `0.5754313094571646`, `0.5633705139683869`, `0.5731185912680369`; every fold improves. Run `31485990777`, source `cb350b4e7ba6bb4a6d283f91bae4d6ea13235d57`, metrics artifact `9100228112`.

**Consequence:** Current honest development benchmark becomes `0.5683065131240066`. Future combinations should treat explicit attribute identity as a retained strong signal and should not collapse it back to only aggregate agreement/conflict ratios.
