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

## D034 — The submission timeout was a CPU-utilization bug, not a model problem

**Decision:** Treat the platform timeouts as an inference-engineering defect and fix them with prediction-preserving changes. Do not trade quality for speed until measured runtime says it is necessary.

**Evidence:** profiling the real structured path on 90,000 synthetic Ozon-shaped pairs showed `2210.1us` per pair, single-threaded. Projected onto the organizer host that is `~254s` of the `360s` public budget and `~608s` of the `780s` private budget, for the structured phase alone, before load, neural inference, meta and write. The solution used one of twenty available CPU cores.

**What changed:** shared `difflib` results between the legacy and typed passes; structured chunks scored in a `fork` worker pool at unchanged chunk boundaries; single-pass `select_items_by_ids`; one shared `ItemNorm` pass behind the contrastive and teacher text caches. Result `487.0us` per pair, `4.54x` on a 10-core host, projected `~22s` public and `~44s` private.

**Safety:** all three paths produce bitwise identical score vectors for all four structured signals, asserted by tests that drive the real `_structured_scores_streaming`. Strict OOF therefore remains `0.6006003614522999` by construction, not by re-measurement.

**Binding lesson:** a fixed-overhead smoke is not runtime evidence. The 64-row CPU smoke reported `24.14s`, of which almost everything was model loading; it could not have revealed a per-pair cost problem. Any future runtime claim must come from a run whose pair count is within an order of magnitude of the private test set.

## D035 — Structured chunk size is pinned, and parallelism must not change it

**Decision:** `STRUCTURED_CHUNK_SIZE` stays at `10_000`. Parallelism is implemented by distributing the existing chunks, never by re-chunking.

**Evidence:** `predict_proba` runs float32 GEMM whose accumulation order depends on the row count of the call. Rescoring the same pairs with a different chunk size moved scores by `~3e-8`. Irrelevant to ranking, but it breaks byte-reproducible packaging and would silently invalidate a hash-verified archive. A test pins the constant and a second test asserts the perturbation stays within float32 noise.

## D036 — Do not replace the global percentile-rank fusion with per-category ranking

**Decision:** Keep `0.5*percentile_rank(category_shrunk) + 0.5*percentile_rank(hgb)` with ranks over the entire scored batch.

**Evidence:** the fusion is batch-dependent, because validation ranks over all `285,210` development rows at once while the platform ranks `~115,000` public and `~275,000` private rows in separate runs. Simulating that over 8 seeds with 20 categories of varying prevalence and separability, holding rows, labels and raw signals fixed: worst `|delta macro AP|` is `0.000257` for global ranking and `0.002812` for per-category ranking. Per-category ranking is worse because small per-category batches quantize the transform.

**Consequence:** roughly `+/-0.0003` of local-to-leaderboard wobble is irreducible under this architecture. The retained v6 margin over `0.6000` is `0.0006`, so a public score slightly below `0.6000` is consistent with an honest local `0.6006` and is not by itself evidence of a broken pipeline.

## D037 — The submission archive file list is derived from the import graph

**Decision:** `ecup_matching/ci/runtime_closure.py` computes the first-party import closure of the v6 entrypoint; the final-submit workflow copies and then verifies that closure instead of a hand-maintained `cp` list.

**Evidence:** the manual list in `ecup-v6-final-submit.yml` did not include the new `v6_parallel.py`, and also did not overwrite `data_subset.py` or `predict_v5.py`, both of which the v5 base archive ships and both of which this iteration changed. The first failure mode is a `ModuleNotFoundError` on the platform; the second is worse, because the archive would silently run stale v5 code and produce different predictions than the ones validated. Tests assert the closure is complete, that the workflow uses the derived list, and that no training-only module is reachable from the runtime entrypoint.

## D038 — A diagnostic driver must persist its model or the result is unusable

**Decision:** Any GPU driver whose result can select an architecture must call `save_pretrained` before exiting.

**Evidence:** `run_v7_fold0_probe.py` writes only `v7-fold0-probe-oof.parquet` and `metrics.json`; it contains no `save_pretrained`, `torch.save` or `state_dict`. The checkpoints that scored `0.6791967999009738` and later `0.7023556010133556` on fold 0 existed only in VRAM and are gone. Building any v7 archive therefore required a full retrain first, at roughly an hour of RTX 2060 time per attempt.

**Consequence:** `run_v7_production.py` saves weights and asserts a `.safetensors` file exists before writing its report. A test compares the two drivers so the probe cannot silently start saving without the comment explaining why the refit was needed being updated.

## D039 — The v7 archive imports inference code through `v7_runtime`, never `v7_neural`

**Decision:** `predict_v7` imports `build_v7_text_cache_from_parquet` and `predict_pairs` from `ecup_matching.ml.v7_runtime`. Both training and inference import that one module, so serialization and scoring cannot drift.

**Evidence:** `v7_neural` imports `train_v5_teacher_fold` and `v5_teacher2_objective`, which transitively reach `split`, `metrics`, `v5_evaluation`, `v5_validation`, `train_v1`, `reranker_data`, `v5_contrastive_data` and more. The v7 runtime closure was `26` modules with those training-only files inside it; after the split it is `8` with none. An earlier iteration already lost an organizer smoke to exactly this class of training-only import.

## D040 — A production refit may never be labeled with a fold diagnostic

**Decision:** `model_v7_metadata.json` records `diagnostic_fold0_macro_average_precision` and `diagnostic_fold0_is_not_out_of_fold: true`, and leaves `strict_oof_macro_average_precision` null until the five-fold outer OOF driver produces one. `build_model_metadata` raises if both fields are set to the same value, and `validate_v7_metadata` repeats the check at container start.

**Reason:** the fold-0 number scores a model trained on folds 1-4 and evaluated on fold 0. The production model is refit on all `285,210` development rows, so no fold can score it. Writing the diagnostic into the strict field would state a validated quality the archive does not have.

**Status:** neither `ecup-v7-full-oof-1ep.yml` nor `ecup-v7-full-oof-fastinfer.yml` has ever executed, so v7 still has no honest out-of-fold number.

## D041 — gpu-dispatch has no secrets, so the archive is built on the runner

**Decision:** `ecup-v7-production.yml` trains, packages and organizer-smokes on `ecup-rtx2060`, then uploads the finished ZIP as a dispatch artifact.

**Evidence:** `repos/MakSoS1/gpu-dispatch/actions/secrets` is empty, and run `31530503704` failed with `HF_TOKEN secret missing in gpu-dispatch`. Without a token there is no way to move roughly `700 MB` of weights into the packaging repository, and no cross-repository PAT exists for Actions artifacts either. Building in place removes the dependency entirely; the HF upload step stays but is skipped while the secret is absent.
