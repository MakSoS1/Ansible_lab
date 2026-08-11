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

**Incident:** runs `31481012401` and `31482891498` failed before ingest because their memory-triggering commits landed during intentionally RED TDD; later GREEN code did not retroactively checkpoint.

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

**Decision:** `ecup_matching/experiments/CURRENT.json` and `ecup_matching/experiments/v*/SAFE_METRICS.json` must be part of `scripts/memory_ingest.py::canonical_sources()`.

**Reason:** Before this audit those files could be correct in Git but invisible to semantic Memora retrieval. A regression test requires their inclusion.

## D027 — Explicit per-key attribute features are retained

**Decision:** Keep fold-trained explicit per-key attribute match/conflict/missing features inside category specialists. This is distinct from rejected direct attribute likelihood score shifting: the estimator decides when a key matters instead of receiving an unconditional logit correction.

**Evidence:** OOF `0.5683065131240066` vs category base `0.5476780661335778`, delta `+0.02062844699042876`; every fold improved. Run `31485990777`.

## D028 — Six heterogeneous signals are retained as the v5 inference base

**Decision:** Retain weak, sparse, explicit, supervised contrastive, pair-teacher and typed-explicit scores as separate inference signals and combine them only after target-free percentile-rank normalization.

**Evidence:** equal six-signal strict OOF Macro AP `0.5975445721449741`, above the preceding five-signal teacher blend `0.5952697490140912`. The byte-verified submission is preserved as fallback at private HF `submissions/v5/0.5975445721`; its competition ZIP SHA-256 is `ee6fec40fe7e79095c33b5a2ed8a1c6cb40e01c3a8e90850c7459d5f1afad06e`.

## D029 — Target-fitted meta layers must be outer-cross-fitted

**Decision:** Any blender that learns from development labels must predict each outer fold using parameters fit without labels from that fold. Hyperparameter selection must also stay inside outer-train where applicable.

**Evidence:** strict global simplex meta OOF `0.5992720660193247`; fully nested category-logistic OOF `0.5988060044248327`. The latter was not retained standalone. Full-fit diagnostic scores are never reported as validation.

## D030 — Fixed category-shrunk simplex crosses the 0.60 engineering milestone

**Decision:** Retain the predeclared category-shrunk simplex: outer-train global simplex plus outer-train local per-category simplex with fixed shrinkage `(support*local + 8000*global)/(support+8000)`.

**Evidence:** strict outer-OOF Macro AP **`0.60095424180184`**, reproduced exactly after refactor; sealed gold remained unopened. Evidence run `31524781399`, artifact `9114649149`, private HF `experiments/v5/category-shrunk/efa629cc0435`.

**Production fallback:** a separately organizer-smoked package is stored at private HF `submissions/v5/0.6009542418`; competition ZIP SHA-256 `3a5341c42346727793ab8877ee6bc8f07e3ac4f18f97c32a9d39d76b5e0609c1`, Actions artifact `9114889240`. It passed exact offline organizer-image smoke and full tests (`225 passed`, one warning).

## D031 — Fixed HGB is complementary but not the standalone best

**Decision:** Retain the fixed shallow HGB OOF vector as a complementary meta signal, not as standalone production best.

**Evidence:** fully outer-cross-fitted fixed HGB Macro AP `0.6006290884983169`, below category-shrunk `0.60095424180184`. No HGB hyperparameter grid was fit to this result. Private HF `experiments/v5/hgb-meta-stack/84a934484619`.

## D032 — Current honest development best is fixed 50/50 category-shrunk + HGB rank fusion

**Decision:** The current v5 development best is exactly `0.5*percentile_rank(category_shrunk_oof) + 0.5*percentile_rank(hgb_stack_oof)`. The 50/50 formula was frozen before the fusion metric was inspected; no alternative weight was searched after the result.

**Evidence:** strict outer-OOF Macro AP **`0.6018115534135564`**, delta `+0.000857311611716427` vs category-shrunk alone. It improved category-shrunk on every outer fold: `0.6003179540`, `0.6073630563`, `0.6122052716`, `0.5973819202`, `0.6105222736`. Evidence run `31525549063`, artifact `9114783508`, private HF `experiments/v5/category-hgb-fusion/79de99434912`.

**Safety state:** sealed gold remains unopened and `0` gold rows have been scored. Public/private leaderboard performance remains unknown until an actual platform submission; local OOF must not be relabeled as leaderboard evidence.

## D033 — The 0.6018115534 model is now the verified production submission

**Decision:** Promote v5 category-shrunk + HGB equal-rank fusion from development-best to verified production package. The exact competition ZIP to submit is `ecup-v5-category-hgb-fusion-0.6018115534-submission.zip` from private HF `submissions/v5/0.6018115534`.

**Binary evidence:** competition ZIP SHA-256 `442769bd2c92d43730d7034fb91d8a83e596a8445ae3c3f887783890e90284d5`; final workflow run `31526323018`, job `93895429369`; Actions artifact `9116032675`, artifact digest `fc6a72f63146df414c5ff4de4aef62a4568e516a12d465830492941348824a46`.

**Runtime evidence:** exact organizer image `odsai/ecup26-matching-baseline:1.0`; CI and organizer sklearn both `1.9.0`; HGB joblib loaded inside organizer image; full offline `run.py` smoke with `--network none` and read-only filesystem passed; output schema/row count/finite nonconstant predictions passed; full repository suite after smoke `230 passed, 1 warning`.

**Consequence:** v2 is only historical hidden evidence, not current production best. The retained v5 fallbacks are category-shrunk `0.6009542418` and six-signal `0.5975445721`. Public/private leaderboard AP remains unknown until platform submission and must be stored separately from strict local OOF.
