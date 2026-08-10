# E-CUP Matching — Experiment Index

This file is the short registry. Detailed metrics live in each experiment's `RESULTS.md`.

| Version | Status | Model / idea | Validation | Macro AP | Runtime / execution evidence | Private artifact | Next decision |
|---|---|---|---|---:|---|---|---|
| v1 | completed | structured/lexical HGB | item-disjoint connected-component split, 73,131 pairs | 0.4961654895 | 308.57 s train/eval; 1.78 s / 1k-pair organizer smoke | `submissions/v1/ecup-v1-submission.zip` | superseded by v2b |
| v2 | completed | 2024-inspired product features + confidence-filtered LLM weak labels (`v2b-weak-curriculum`) | exact same item-disjoint validation, 73,131 pairs, 0 shared items | **0.5010008995** | 334 s / 275k-pair offline organizer benchmark; 446 s headroom to 780 s private limit | `submissions/v2/ecup-v2-submission.zip` | superseded by retained v3; remains fast anchor/fallback |
| v3 | **completed / retained** | v2b structured anchor + `rubert-tiny2` stage-1 global reranker blend, neural alpha 0.45; real model-mined hard-negative stage-2 ablation | exact same item-disjoint validation, 73,131 pairs, 0 shared items | **0.5254642646** | GitHub M1 MPS training; exact organizer-image `--network none` canonical smoke: 10k/10k neural pairs, 103 s CPU wall; 1k independent neural smoke also green | `submissions/v3/ecup-v3-submission.zip` | current best; start v4 only for a demonstrably better item-disjoint ablation |

## v2 ablation headline

- v2a, human + 2024-inspired product-aware features: `0.5006971263` Macro AP — accepted.
- v2b, v2a + 300k confidence-filtered weak labels: `0.5010008995` — retained at the time, now superseded by v3.
- v2c, naive static hard-negative reweighting: `0.4957263069` — rejected.
- Lightning neural reranker code was implemented, but Studio/direct Job allocation was denied by the authenticated account; no neural metric was fabricated from those failed probes.

## v3 headline

- Free GPU actually used: standard GitHub-hosted `macos-15` Apple M1 with PyTorch MPS.
- Human-only compact neural train: 180,000 rows; fixed validation unchanged at 73,131 rows with 0 shared item IDs.
- Production neural run: 1,600 stage-1 steps + 300 model-mined-hard-negative stage-2 steps.
- Hard-negative mining scored 180,000 authoritative human rows and selected 12,000 hard negatives; the stage-2 checkpoint was evaluated and rejected by the fixed validation.
- Selected candidate: `stage1-global`, v2b/neural weights `0.55 / 0.45`.
- Macro AP: **`0.5254642645846543`**, absolute delta vs v2b **`+0.024463365088784106`** (~4.88% relative).
- Canonical source tests: **108 passed**; memory policy PASS.
- Canonical ZIP: `109,185,253` bytes; SHA-256 `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`.
- Canonical exact-image offline smoke: 10,000 pairs, **10,000 neural pairs**, output verified, network disabled, private HF upload verified.
- A category-routing normalization defect was discovered by a rejected sprint package (`neural_pairs=0`) and fixed RED/GREEN before the canonical v3 ZIP was published.

## Required interpretation

- Headline scores are comparable because retained iterations use the identical leakage-resistant human validation split.
- The competition metric is ranking Average Precision by category; threshold accuracy is not the retention criterion.
- The 10k v3 CPU timing is a correctness/stress measurement only because the GitHub Ubuntu runner does not expose NVIDIA CUDA; the organizer image is CUDA-capable and the target evaluation GPU is H100.
- Private artifacts, raw data and learned weights never belong in this public Git repository.
