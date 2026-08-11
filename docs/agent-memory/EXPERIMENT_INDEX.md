# E-CUP Matching — Experiment Index

This file is the short registry. Detailed metrics live in each experiment's `RESULTS.md`.

| Version | Status | Model / idea | Validation | Macro AP | Runtime / execution evidence | Private artifact | Next decision |
|---|---|---|---|---:|---|---|---|
| v1 | completed | structured/lexical HGB | item-disjoint connected-component split, 73,131 pairs | 0.4961654895 | 308.57 s train/eval; 1.78 s / 1k-pair organizer smoke | `submissions/v1/ecup-v1-submission.zip` | superseded by v2b |
| v2 | completed | 2024-inspired product features + confidence-filtered LLM weak labels (`v2b-weak-curriculum`) | exact same item-disjoint validation, 73,131 pairs, 0 shared items | **0.5010008995** | 334 s / 275k-pair offline organizer benchmark; 446 s headroom to 780 s private limit | `submissions/v2/ecup-v2-submission.zip` | superseded by retained v3; remains fast anchor/fallback |
| v3 | completed / retained fallback | v2b structured anchor + `rubert-tiny2` stage-1 global reranker blend, neural alpha 0.45 | exact same item-disjoint validation, 73,131 pairs, 0 shared items | **0.5254642646** | exact organizer-image offline canonical smoke: 10k/10k neural pairs, network disabled | canonical SHA `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660` | immutable fallback; superseded by v4 |
| v4 | **completed / retained current best** | immutable v3 structured+neural scores with 5-fold component-cross-fitted, shrinkage-regularized per-category neural alphas; selected prior 4000 | same 73,131 rows; 0 train/validation item overlap; 53,131 validation item-components for routing cross-fit | **0.5276431099 OOF**; 0.5284493943 full-fit coefficients | exact organizer image / `--network none`: 1k/1k neural pairs, valid output; canonical private freeze PASS | canonical SHA `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849` | use v4 for submission; keep v3 fallback; stronger ruBERT ladder becomes future ablation |

## v2 ablation headline

- v2a, human + 2024-inspired product-aware features: `0.5006971263` Macro AP — accepted.
- v2b, v2a + 300k confidence-filtered weak labels: `0.5010008995` — retained at the time, now superseded by v3/v4.
- v2c, naive static hard-negative reweighting: `0.4957263069` — rejected.
- Lightning neural reranker code was implemented, but Studio/direct Job allocation was denied by the authenticated account; no neural metric was fabricated from those failed probes.

## v3 headline

- Free GPU actually used: standard GitHub-hosted `macos-15` Apple M1 with PyTorch MPS.
- Human-only compact neural train: 180,000 rows; fixed validation unchanged at 73,131 rows with 0 shared item IDs.
- Production neural run: 1,600 stage-1 steps + 300 model-mined-hard-negative stage-2 steps.
- Hard-negative mining scored 180,000 authoritative human rows and selected 12,000 hard negatives; the stage-2 checkpoint was evaluated and rejected by the fixed validation.
- Selected candidate: `stage1-global`, v2b/neural weights `0.55 / 0.45`.
- Macro AP: `0.5254642645846543`, absolute delta vs v2b `+0.024463365088784106` (~4.88% relative).
- Canonical ZIP: `109,185,253` bytes; SHA-256 `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`.
- Canonical exact-image offline smoke: 10,000 pairs, 10,000 neural pairs, output verified, network disabled, private HF upload verified.
- A category-routing normalization defect was discovered by a rejected sprint package (`neural_pairs=0`) and fixed RED/GREEN before the canonical v3 ZIP was published.

## v4 headline

- Retained architecture does **not** claim a newly trained `ruBert-base`: it preserves the canonical v3 v2b + `rubert-tiny2` models and improves only their category-conditioned score combination.
- Routing selection uses 5-fold `GroupKFold` over connected components of **all validation candidate edges**, so a held-out component never tunes its own alpha.
- Cross-fitted global blend: `0.526005894031544`.
- Selected shrinkage prior: `4000` from candidate set `250, 500, 1000, 2000, 4000, 8000`.
- Honest cross-fitted Macro AP: **`0.5276431099433088`**, delta vs v3 **`+0.0021788453586544243`**.
- After prior selection, deployable alphas are fit on all labelled validation rows; their full-fit score is `0.5284493942551521`. This is package-coefficient evidence, not the headline OOF estimate.
- Freeze run: `31474888023`; cross-fit selection run: `31473553650`.
- Frozen validation predictions SHA-256: `4112aa2556cb683ffca27cd9bd16c00a7149bb7e3279d1f2a6abb2b20438d643`.
- Canonical v4 ZIP: `109,185,879` bytes; SHA-256 `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`.
- Exact organizer-image offline smoke: 1,000 rows, **1,000 neural pairs**, 1,000 unique scores, output/order/range checks PASS, network disabled.
- The originally planned stronger `ai-forever/ruBert-base` ladder remains implemented but unretained. Its first RTX attempt died before metrics during a host-memory spike; the data path was subsequently made bounded-memory. No ruBERT metric is mixed into retained v4 evidence.

## Required interpretation

- v4's **cross-fitted** routing score is the retained quality headline; the full-fit score is not presented as unbiased generalization evidence.
- The competition metric is ranking Average Precision by category; threshold accuracy is not the retention criterion.
- v3 remains an immutable fallback and provides the actual learned model weights used by v4; v4 changes the deployable category routing coefficients only.
- Private artifacts, raw data and learned weights never belong in this public Git repository.